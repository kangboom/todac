"""
지식 베이스 서비스 (문서 업로드, 조회, 삭제)
"""
from sqlalchemy.orm import Session
from fastapi import HTTPException, status, UploadFile
from app.models.knowledge import KnowledgeDoc
from app.core.milvus_schema import get_milvus_collection_safe
from app.core.config import settings
from app.services.parsers.llama_parse_parser import LlamaParseParser
from app.services.parsers.pymupdf_parser import PyMuPDFParser
from app.services.parsers.docling_parser import DoclingParser
from app.services.chunking_markdown import chunk_markdown_documents
from app.services.parsers.base import BaseParser
from app.agent.tools import get_embedding
from pymilvus import Collection
from typing import List, Dict, Any, Optional
from io import BytesIO
import uuid
import hashlib
import logging
import json
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# 파서 인스턴스 (설정에 따라 선택)
_parsers: List[BaseParser] = []


def _initialize_parsers():
    """설정에 따라 파서 초기화"""
    global _parsers
    if _parsers:
        return  # 이미 초기화됨
    
    parser_type = settings.PDF_PARSER.lower()
    
    if parser_type == "llamaparse":
        try:
            _parsers.append(LlamaParseParser())
            logger.info("LlamaParse 파서가 초기화되었습니다.")
        except Exception as e:
            logger.error(f"LlamaParse 파서 초기화 실패: {str(e)}")
            # 폴백으로 PyMuPDF 시도
            try:
                _parsers.append(PyMuPDFParser())
                logger.warning("LlamaParse 파서 초기화 실패, PyMuPDF 파서로 폴백합니다.")
            except Exception as e2:
                logger.error(f"PyMuPDF 파서 초기화도 실패: {str(e2)}")
                raise ValueError("PDF 파서를 초기화할 수 없습니다. 필요한 패키지가 설치되어 있는지 확인하세요.")
    
    elif parser_type == "pymupdf":
        try:
            _parsers.append(PyMuPDFParser())
            logger.info("PyMuPDF 파서가 초기화되었습니다.")
        except Exception as e:
            logger.error(f"PyMuPDF 파서 초기화 실패: {str(e)}")
            # 폴백으로 LlamaParse 시도 (API 키가 있는 경우)
            if settings.LLAMAPARSE_API_KEY:
                try:
                    _parsers.append(LlamaParseParser())
                    logger.warning("PyMuPDF 파서 초기화 실패, LlamaParse 파서로 폴백합니다.")
                except Exception as e2:
                    logger.error(f"LlamaParse 파서 초기화도 실패: {str(e2)}")
                    raise ValueError("PDF 파서를 초기화할 수 없습니다. 필요한 패키지가 설치되어 있는지 확인하세요.")
            else:
                raise ValueError(
                    "PyMuPDF 파서를 초기화할 수 없습니다. "
                    "pymupdf와 pymupdf4llm 패키지가 설치되어 있는지 확인하거나, "
                    "LlamaParse를 사용하려면 LLAMAPARSE_API_KEY를 설정하세요."
                )
    
    elif parser_type == "docling":
        try:
            _parsers.append(DoclingParser())
            logger.info("Docling 파서가 초기화되었습니다.")
        except Exception as e:
            logger.error(f"Docling 파서 초기화 실패: {str(e)}")
            # 폴백으로 PyMuPDF 시도
            try:
                _parsers.append(PyMuPDFParser())
                logger.warning("Docling 파서 초기화 실패, PyMuPDF 파서로 폴백합니다.")
            except Exception as e2:
                logger.error(f"PyMuPDF 파서 초기화도 실패: {str(e2)}")
                # 폴백으로 LlamaParse 시도 (API 키가 있는 경우)
                if settings.LLAMAPARSE_API_KEY:
                    try:
                        _parsers.append(LlamaParseParser())
                        logger.warning("PyMuPDF 파서 초기화도 실패, LlamaParse 파서로 폴백합니다.")
                    except Exception as e3:
                        logger.error(f"LlamaParse 파서 초기화도 실패: {str(e3)}")
                        raise ValueError("PDF 파서를 초기화할 수 없습니다. 필요한 패키지가 설치되어 있는지 확인하세요.")
                else:
                    raise ValueError("PDF 파서를 초기화할 수 없습니다. 필요한 패키지가 설치되어 있는지 확인하세요.")
    else:
        raise ValueError(
            f"지원하지 않는 PDF 파서 타입: {parser_type}. "
            f"'pymupdf', 'llamaparse', 또는 'docling'을 사용하세요."
        )


def _get_parser(filename: str) -> Optional[BaseParser]:
    """파일명에 맞는 파서 찾기"""
    # 파서가 초기화되지 않았으면 초기화
    if not _parsers:
        _initialize_parsers()
    
    for parser in _parsers:
        if parser.can_parse(filename):
            return parser
    return None


def _calculate_hash(content: bytes) -> str:
    """문서 내용의 SHA-256 해시값 계산"""
    return hashlib.sha256(content).hexdigest()


# S3 클라이언트 싱글톤
_s3_client = None

def get_s3_client():
    """S3 클라이언트 싱글톤"""
    global _s3_client
    if _s3_client is None:
        s3_config = {
            'service_name': 's3',
            'region_name': settings.S3_REGION,
        }
        
        # AWS 자격 증명이 있으면 사용
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            s3_config['aws_access_key_id'] = settings.AWS_ACCESS_KEY_ID
            s3_config['aws_secret_access_key'] = settings.AWS_SECRET_ACCESS_KEY
        
        # MinIO 등 로컬 S3 호환 서비스 사용 시
        if settings.S3_ENDPOINT_URL:
            s3_config['endpoint_url'] = settings.S3_ENDPOINT_URL
        
        _s3_client = boto3.client(**s3_config)
    return _s3_client


def _upload_to_s3(content: bytes, s3_key: str, content_type: str = None) -> str:
    """
    S3에 파일 업로드
    
    Args:
        content: 파일 내용 (bytes)
        s3_key: S3 객체 키 (경로)
        content_type: Content-Type (예: 'application/pdf', 'text/markdown')
    
    Returns:
        S3 URL (s3://bucket-name/key 형식)
    """
    try:
        s3_client = get_s3_client()
        bucket_name = settings.S3_BUCKET_NAME
        
        # 업로드 옵션
        upload_kwargs = {
            'Bucket': bucket_name,
            'Key': s3_key,
            'Body': content,
        }
        
        # Content-Type 설정
        if content_type:
            upload_kwargs['ContentType'] = content_type
        else:
            # 파일 확장자로 자동 판단
            if s3_key.endswith('.pdf'):
                upload_kwargs['ContentType'] = 'application/pdf'
            elif s3_key.endswith('.md'):
                upload_kwargs['ContentType'] = 'text/markdown'
            elif s3_key.endswith('.png'):
                upload_kwargs['ContentType'] = 'image/png'
            elif s3_key.endswith('.jpg') or s3_key.endswith('.jpeg'):
                upload_kwargs['ContentType'] = 'image/jpeg'
        
        # S3에 업로드
        s3_client.put_object(**upload_kwargs)
        
        s3_url = f"s3://{bucket_name}/{s3_key}"
        logger.info(f"S3 업로드 완료: {s3_url}")
        
        return s3_url
        
    except ClientError as e:
        logger.error(f"버킷 이름: {bucket_name}")
        logger.error(f"S3 업로드 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"S3 업로드 중 오류가 발생했습니다: {str(e)}"
        )


def _parse_s3_url(s3_url: str) -> tuple[str, str]:
    """
    S3 URL에서 bucket과 key 추출
    
    Args:
        s3_url: S3 URL (s3://bucket-name/key 형식)
    
    Returns:
        (bucket_name, key) 튜플
    """
    if not s3_url.startswith('s3://'):
        raise ValueError(f"유효하지 않은 S3 URL 형식: {s3_url}")
    
    # s3:// 제거
    path = s3_url[5:]
    
    # 첫 번째 '/'를 기준으로 bucket과 key 분리
    parts = path.split('/', 1)
    bucket_name = parts[0]
    key = parts[1] if len(parts) > 1 else ''
    
    return bucket_name, key


def _delete_from_s3(s3_url: str) -> None:
    """
    S3에서 파일 삭제
    
    Args:
        s3_url: S3 URL (s3://bucket-name/key 형식)
    """
    try:
        if not s3_url:
            logger.warning("S3 URL이 비어있어 삭제를 건너뜁니다.")
            return
        
        s3_client = get_s3_client()
        bucket_name, key = _parse_s3_url(s3_url)
        
        # S3에서 파일 삭제
        s3_client.delete_object(Bucket=bucket_name, Key=key)
        
        logger.info(f"S3 파일 삭제 완료: {s3_url}")
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            logger.warning(f"S3 파일이 이미 존재하지 않습니다: {s3_url}")
        else:
            logger.error(f"S3 파일 삭제 실패: {str(e)}, URL: {s3_url}")
            # S3 삭제 실패해도 계속 진행 (PostgreSQL, Milvus 삭제는 진행)
    except Exception as e:
        logger.error(f"S3 파일 삭제 중 예외 발생: {str(e)}, URL: {s3_url}")
        # 예외 발생해도 계속 진행


def _generate_storage_paths(doc_id: uuid.UUID, original_filename: str) -> Dict[str, str]:
    """
    S3 저장 경로 생성 (새로운 구조)
    
    Args:
        doc_id: 문서 ID
        original_filename: 원본 파일명
    
    Returns:
        {
            'raw_pdf_key': 'raw/doc_12345/original.pdf',
            'processed_md_key': 'processed/doc_12345/content.md',
            'images_dir': 'processed/doc_12345/images/'
        }
    """
    doc_dir = f"doc_{doc_id}"
    
    # 원본 파일명에서 확장자 제거
    base_filename = original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename
    
    return {
        'raw_pdf_key': f"raw/{doc_dir}/{original_filename}",
        'processed_md_key': f"processed/{doc_dir}/{base_filename}.md",
        'images_dir': f"processed/{doc_dir}/images/"
    }


class KnowledgeService:
    """지식 베이스 서비스"""
    
    @staticmethod
    def ingest_document(
        db: Session,
        file: UploadFile,
        category: str,
        user_id: uuid.UUID
    ) -> KnowledgeDoc:
        """
        문서 업로드 및 벡터 DB 저장
        
        Args:
            db: 데이터베이스 세션
            file: 업로드된 파일
            category: 문서 카테고리
            user_id: 업로드한 사용자 ID
        
        Returns:
            생성된 KnowledgeDoc
        """
        # 1. 파일 읽기
        content = file.file.read()
        filename = file.filename
        
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="파일 내용이 비어있습니다."
            )
        
        # 2. 문서 해시 계산 (중복 업로드 체크)
        doc_hash = _calculate_hash(content)
        
        # 중복 문서 확인
        existing_doc = db.query(KnowledgeDoc).filter(KnowledgeDoc.doc_hash == doc_hash).first()
        if existing_doc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"동일한 문서가 이미 업로드되어 있습니다. (문서 ID: {existing_doc.id})"
            )
        
        # 3. 파서 찾기
        parser = _get_parser(filename)
        if not parser:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"지원하지 않는 파일 형식입니다. 지원 형식: PDF"
            )
        
        # 4. 문서 파싱
        try:
            documents = parser.parse(content, filename)
        except Exception as e:
            logger.error(f"문서 파싱 실패: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"문서 파싱 중 오류가 발생했습니다: {str(e)}"
            )
        
        # 5. 텍스트 청킹 (Markdown인 경우 MarkdownHeaderTextSplitter 사용)
        # PDF 파서는 모두 Markdown을 반환하므로 헤더 기반 청킹 사용
        if isinstance(parser, (LlamaParseParser, PyMuPDFParser, DoclingParser)):
            # Markdown 문서는 헤더 기반 청킹
            chunks = chunk_markdown_documents(documents)
        else:
            # 일반 텍스트는 기존 방식 사용
            from app.services.chunking import chunk_documents
            chunks = chunk_documents(documents)
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="파싱된 텍스트가 없습니다."
            )
        
        # 6. 문서 ID 생성
        doc_id = uuid.uuid4()
        
        # 7. Markdown 텍스트 추출 (S3 업로드용)
        markdown_text = documents[0]["text"] if documents else ""
        markdown_bytes = markdown_text.encode('utf-8')
        
        # 8. S3 저장 경로 생성
        storage_paths = _generate_storage_paths(doc_id, filename)
        
        # 9. S3에 파일 업로드
        # 원본 PDF 업로드 (raw/)
        raw_pdf_url = _upload_to_s3(
            content=content,
            s3_key=storage_paths['raw_pdf_key'],
            content_type='application/pdf'
        )
        
        # 변환된 Markdown 업로드 (processed/)
        storage_url = _upload_to_s3(
            content=markdown_bytes,
            s3_key=storage_paths['processed_md_key'],
            content_type='text/markdown'
        )
        
        # TODO: 이미지 추출 및 업로드 (옵션)
        # PDF에서 이미지를 추출하는 경우:
        # images = extract_images_from_pdf(content)  # 별도 함수 필요
        # for idx, image_data in enumerate(images):
        #     image_key = f"{storage_paths['images_dir']}img{idx+1}.png"
        #     _upload_to_s3(image_data, image_key, content_type='image/png')
        
        # 10. 임베딩 생성 및 Milvus 저장
        try:
            collection = get_milvus_collection_safe()
            
            # 임베딩 생성
            embeddings = []
            milvus_data = []
            
            for chunk in chunks:
                embedding = get_embedding(chunk["text"])
                embeddings.append(embedding)
                
                # 헤더 정보 추출 (Header 1, Header 2, Header 3 등)
                header_metadata = {
                    k: v for k, v in chunk.get("metadata", {}).items() 
                    if k.startswith("Header")
                }
                headers_json = json.dumps(header_metadata, ensure_ascii=False) if header_metadata else "{}"
                
                milvus_data.append({
                    "doc_id": str(doc_id),
                    "chunk_index": chunk["chunk_index"],
                    "embedding": embedding,
                    "content": chunk["text"][:65535],  # Milvus VARCHAR 제한
                    "filename": filename[:255],
                    "category": category[:50],
                    "headers": headers_json[:2048]  # Milvus VARCHAR 제한
                })
            
            # Milvus에 삽입
            collection.insert(milvus_data)
            collection.flush()
            
            logger.info(f"Milvus 저장 완료: {len(milvus_data)}개 청크, doc_id={doc_id}")
            
        except Exception as e:
            logger.error(f"Milvus 저장 실패: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"벡터 DB 저장 중 오류가 발생했습니다: {str(e)}"
            )
        
        # 11. PostgreSQL에 메타데이터 저장
        meta_info = {
            "category": category,
            "uploaded_by": str(user_id),
            "chunk_count": len(chunks),
            "original_filename": filename
        }
        
        knowledge_doc = KnowledgeDoc(
            id=doc_id,
            filename=filename,
            storage_url=storage_url,
            raw_pdf_url=raw_pdf_url,
            doc_hash=doc_hash,
            meta_info=meta_info
        )
        
        db.add(knowledge_doc)
        db.commit()
        db.refresh(knowledge_doc)
        
        logger.info(f"문서 업로드 완료: doc_id={doc_id}, filename={filename}, chunks={len(chunks)}")
        
        return knowledge_doc
    
    @staticmethod
    def get_documents(
        db: Session,
        category: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[KnowledgeDoc]:
        """
        문서 목록 조회
        
        Args:
            db: 데이터베이스 세션
            category: 카테고리 필터 (선택) - meta_info에서 조회
            limit: 최대 개수
            offset: 오프셋
        
        Returns:
            KnowledgeDoc 리스트
        """
        query = db.query(KnowledgeDoc)
        
        if category:
            # meta_info JSONB 필드에서 category 필터링
            query = query.filter(KnowledgeDoc.meta_info['category'].astext == category)
        
        documents = query.order_by(KnowledgeDoc.created_at.desc()).offset(offset).limit(limit).all()
        
        return documents
    
    @staticmethod
    def delete_document(
        db: Session,
        doc_id: uuid.UUID
    ) -> None:
        """
        문서 삭제 (PostgreSQL + Milvus + S3)
        
        Args:
            db: 데이터베이스 세션
            doc_id: 문서 ID
        """
        # 1. PostgreSQL에서 문서 조회
        knowledge_doc = db.query(KnowledgeDoc).filter(KnowledgeDoc.id == doc_id).first()
        
        if not knowledge_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="문서를 찾을 수 없습니다."
            )
        
        # 2. S3에서 파일 삭제 (원본 PDF와 변환된 Markdown)
        # storage_url (Markdown 파일) 삭제
        if knowledge_doc.storage_url:
            _delete_from_s3(knowledge_doc.storage_url)
        
        # raw_pdf_url (원본 PDF 파일) 삭제
        if knowledge_doc.raw_pdf_url:
            _delete_from_s3(knowledge_doc.raw_pdf_url)
        
        # TODO: 이미지 파일들도 삭제 (processed/doc_{doc_id}/images/ 디렉토리)
        # doc_id로 디렉토리 경로를 알 수 있으므로 해당 디렉토리의 모든 파일 삭제 가능
        
        # 3. Milvus에서 해당 문서의 모든 청크 삭제
        try:
            collection = get_milvus_collection_safe()
            
            # doc_id로 필터링하여 삭제
            collection.delete(expr=f'doc_id == "{doc_id}"')
            collection.flush()
            
            logger.info(f"Milvus에서 문서 삭제 완료: doc_id={doc_id}")
            
        except Exception as e:
            logger.error(f"Milvus 삭제 실패: {str(e)}", exc_info=True)
            # Milvus 삭제 실패해도 PostgreSQL은 삭제 진행
        
        # 4. PostgreSQL에서 삭제
        db.delete(knowledge_doc)
        db.commit()
        
        logger.info(f"문서 삭제 완료: doc_id={doc_id}")


knowledge_service = KnowledgeService()

