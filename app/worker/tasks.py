"""
TaskIQ Worker Tasks
문서 처리 등 백그라운드 작업 정의
"""
import uuid
import logging
import json
import boto3
from typing import List
from app.core.taskiq import broker
from app.core.database import SessionLocal
from app.models.knowledge import KnowledgeDoc
from app.services.parsers.llama_parse_parser import LlamaParseParser
from app.services.parsers.pymupdf_parser import PyMuPDFParser
from app.services.parsers.docling_parser import DoclingParser
from app.services.chunking_markdown import chunk_markdown_documents
from app.services.markdown_service import cleanup_markdown_with_llm
from app.services.parser_service import get_parser
from app.services.s3_service import upload_to_s3, delete_from_s3, generate_storage_paths
from app.agent.tools import get_embedding
from app.core.milvus_schema import get_milvus_collection_safe
from app.core.config import settings

logger = logging.getLogger(__name__)

# S3 클라이언트 (boto3)
s3_client = boto3.client(
    's3',
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.S3_REGION
)

@broker.task
async def process_document_task(
    doc_id_str: str,
    raw_s3_key: str,
    filename: str,
    category: str,
    user_id_str: str,
    file_size: int,
    doc_hash: str
):
    """
    백그라운드 문서 처리 태스크
    1. S3에서 원본 파일 다운로드
    2. 문서 파싱 & Markdown 변환
    3. Markdown S3 업로드
    4. 청킹 & 임베딩
    5. Milvus 저장
    6. DB 메타데이터 저장
    """
    logger.info(f"🚀 문서 처리 태스크 시작: doc_id={doc_id_str}, file={filename}")
    
    db = SessionLocal()
    doc_id = uuid.UUID(doc_id_str)
    user_id = uuid.UUID(user_id_str)
    
    # 롤백용 리소스 추적
    uploaded_s3_keys = []
    milvus_inserted = False
    
    try:
        # 1. S3에서 파일 다운로드
        try:
            response = s3_client.get_object(Bucket=settings.S3_BUCKET_NAME, Key=raw_s3_key)
            content = response['Body'].read()
            logger.info(f"S3 다운로드 완료: {len(content)} bytes")
        except Exception as e:
            logger.error(f"S3 파일 다운로드 실패: {e}")
            raise e

        # 2. 파서 찾기
        parser = get_parser(filename)
        if not parser:
            logger.error(f"지원하지 않는 파일 형식: {filename}")
            return # 실패 처리 (DB 업데이트 등 필요할 수 있음)

        # 3. 문서 파싱
        try:
            documents = parser.parse(content, filename)
        except Exception as e:
            logger.error(f"문서 파싱 실패: {e}")
            raise e
            
        # 3-1. Markdown 보정
        if isinstance(parser, (LlamaParseParser, PyMuPDFParser, DoclingParser)):
            if documents and len(documents) > 0:
                original_text = documents[0].text
                if original_text:
                    cleaned_text = cleanup_markdown_with_llm(original_text, filename)
                    documents[0].text = cleaned_text
                    logger.info(f"Markdown 보정 완료: {filename}")

        # 4. 텍스트 청킹
        chunks = chunk_markdown_documents(documents)
        if not chunks:
            raise ValueError("파싱된 텍스트가 없습니다.")

        # 5. Markdown 텍스트 S3 업로드 (Processed)
        markdown_text = documents[0].text if documents else ""
        markdown_bytes = markdown_text.encode('utf-8')
        
        storage_paths = generate_storage_paths(doc_id, filename)
        
        # processed_md_key로 업로드 (raw_s3_key는 이미 API 서버에서 올림)
        storage_url = upload_to_s3(
            content=markdown_bytes,
            s3_key=storage_paths.processed_md_key,
            content_type='text/markdown'
        )
        uploaded_s3_keys.append(storage_url)

        # 6. 임베딩 및 Milvus 저장
        try:
            collection = get_milvus_collection_safe()
            milvus_data = []
            
            for chunk in chunks:
                header_metadata = {
                    k: v for k, v in chunk.metadata.items() 
                    if k.startswith("Header")
                }
                
                if header_metadata:
                    sorted_headers = [header_metadata[k] for k in sorted(header_metadata.keys())]
                    header_path = " > ".join(sorted_headers)
                    embedding_text = f"{header_path}\n\n{chunk.text}"
                else:
                    embedding_text = chunk.text
                
                embedding = get_embedding(embedding_text)
                
                headers_json = json.dumps(header_metadata, ensure_ascii=False) if header_metadata else "{}"
                
                milvus_data.append({
                    "doc_id": str(doc_id),
                    "chunk_index": chunk.chunk_index,
                    "embedding": embedding,
                    "content": chunk.text[:65535],
                    "filename": filename[:255],
                    "category": category[:50],
                    "headers": headers_json[:2048]
                })
            
            collection.insert(milvus_data)
            collection.flush()
            milvus_inserted = True
            logger.info(f"Milvus 저장 완료: {len(milvus_data)}개 청크")
            
        except Exception as e:
            logger.error(f"Milvus 저장 실패: {e}")
            raise e

        # 7. DB 저장 (KnowledgeDoc)
        # 이미 API에서 raw_pdf_url 등을 알고 있으므로 여기서 최종 저장을 수행
        # raw_pdf_url은 raw_s3_key를 통해 구성하거나 API에서 넘겨받을 수도 있지만,
        # 여기서는 raw_s3_key를 알고 있으니 URL을 구성하거나 S3 서비스 함수 활용
        
        # raw_s3_key 예: raw/uuid/filename.pdf
        # s3_service.upload_to_s3가 반환하는 형식에 맞춰야 함
        # 여기서는 간단히 raw_s3_key를 그대로 사용하거나, 필요한 URL 포맷으로 저장
        
        # upload_to_s3 함수는 전체 URL을 반환하므로, API 서버에서 업로드했을 때 받은 URL을 넘겨받는 게 좋음
        # 하지만 raw_s3_key만 받아도 충분함.
        
        raw_pdf_url = f"https://{settings.S3_BUCKET_NAME}.s3.{settings.S3_REGION}.amazonaws.com/{raw_s3_key}"

        meta_info = {
            "category": category,
            "uploaded_by": str(user_id),
            "chunk_count": len(chunks),
            "original_filename": filename,
            "status": "completed"  # 상태 표시
        }
        
        knowledge_doc = KnowledgeDoc(
            id=doc_id,
            filename=filename,
            storage_url=storage_url,
            raw_pdf_url=raw_pdf_url,
            doc_hash=doc_hash,
            file_size=file_size,
            meta_info=meta_info
        )
        
        db.add(knowledge_doc)
        db.commit()
        
        logger.info(f"✅ 문서 처리 완료: doc_id={doc_id}")

    except Exception as e:
        logger.error(f"태스크 처리 중 오류 발생: {e}")
        
        # 롤백
        db.rollback()
        
        # S3 삭제 (Processed만 삭제, Raw는 남길지 고민 필요하지만 실패 시 다 지우는 게 깔끔)
        # Raw 파일은 태스크 시작 전 API 서버가 올린 것. 실패 시 지워야 함.
        try:
            # Raw 파일 삭제
            raw_url = f"https://{settings.S3_BUCKET_NAME}.s3.{settings.S3_REGION}.amazonaws.com/{raw_s3_key}"
            delete_from_s3(raw_url)
            
            # Processed 파일 삭제
            for url in uploaded_s3_keys:
                delete_from_s3(url)
        except Exception as s3_err:
            logger.error(f"S3 롤백 실패: {s3_err}")
            
        # Milvus 삭제
        if milvus_inserted:
            try:
                collection = get_milvus_collection_safe()
                collection.delete(expr=f'doc_id == "{doc_id_str}"')
                collection.flush()
            except Exception as m_err:
                logger.error(f"Milvus 롤백 실패: {m_err}")
                
        # 실패 상태 DB 기록 등이 필요하다면 여기서 수행 (지금은 생략)
        
    finally:
        db.close()

