"""
지식 베이스 서비스 (문서 업로드, 조회, 삭제)
"""
from sqlalchemy.orm import Session
from fastapi import HTTPException, status, UploadFile
from app.models.knowledge import KnowledgeDoc
from app.core.milvus_schema import get_milvus_collection_safe
from app.services.parsers.llama_parse_parser import LlamaParseParser
from app.services.parsers.pymupdf_parser import PyMuPDFParser
from app.services.parsers.docling_parser import DoclingParser
from app.services.chunking_markdown import chunk_markdown_documents
from app.agent.tools import get_embedding
from app.dto.knowledge import BatchDocumentResult, DocumentResponse, ParsedDocument, Chunk
from app.services.s3_service import upload_to_s3, delete_from_s3, generate_storage_paths
from app.services.markdown_service import cleanup_markdown_with_llm
from app.services.parser_service import get_parser
from typing import List, Optional
import uuid
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


def _calculate_hash(content: bytes) -> str:
    """문서 내용의 SHA-256 해시값 계산"""
    return hashlib.sha256(content).hexdigest()


def ingest_document(
    db: Session,
    file: UploadFile,
    category: str,
    user_id: uuid.UUID
) -> KnowledgeDoc:
    """
    문서 업로드 및 벡터 DB 저장
    """
    # 1. 파일 읽기
    content = file.file.read()
    filename = file.filename
    file_size = len(content)
    
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
    parser = get_parser(filename)
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
    
    # 4-1. Markdown 보정 (PDF 파서인 경우)
    if isinstance(parser, (LlamaParseParser, PyMuPDFParser, DoclingParser)):
        if documents and len(documents) > 0:
            original_text = documents[0].text
            if original_text:
                cleaned_text = cleanup_markdown_with_llm(original_text, filename)
                documents[0].text = cleaned_text
                logger.info(f"Markdown 보정 완료: 파일={filename}")
    
    # 5. 텍스트 청킹 (Markdown인 경우 MarkdownHeaderTextSplitter 사용)
    if isinstance(parser, (LlamaParseParser, PyMuPDFParser, DoclingParser)):
        chunks = chunk_markdown_documents(documents)
    else:
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
    markdown_text = documents[0].text if documents else ""
    markdown_bytes = markdown_text.encode('utf-8')
    
    # 8. S3 저장 경로 생성
    storage_paths = generate_storage_paths(doc_id, filename)
    
    # 9. S3에 파일 업로드
    # 원본 PDF 업로드 (raw/)
    raw_pdf_url = upload_to_s3(
        content=content,
        s3_key=storage_paths['raw_pdf_key'],
        content_type='application/pdf'
    )
    
    # 변환된 Markdown 업로드 (processed/)
    storage_url = upload_to_s3(
        content=markdown_bytes,
        s3_key=storage_paths['processed_md_key'],
        content_type='text/markdown'
    )
    
    # 10. 임베딩 생성 및 Milvus 저장
    try:
        collection = get_milvus_collection_safe()
        
        # 임베딩 생성
        embeddings = []
        milvus_data = []
        
        for chunk in chunks:
            # 헤더 정보 추출 (Header 1, Header 2, Header 3 등)
            header_metadata = {
                k: v for k, v in chunk.metadata.items() 
                if k.startswith("Header")
            }
            
            # 임베딩용 텍스트 생성 (헤더 정보 포함)
            if header_metadata:
                sorted_headers = [
                    header_metadata[k] 
                    for k in sorted(header_metadata.keys())
                ]
                header_path = " > ".join(sorted_headers)
                embedding_text = f"{header_path}\n\n{chunk.text}"
            else:
                embedding_text = chunk.text
            
            embedding = get_embedding(embedding_text)
            embeddings.append(embedding)
            
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
        file_size=file_size,
        meta_info=meta_info
    )
    
    db.add(knowledge_doc)
    db.commit()
    db.refresh(knowledge_doc)
    
    logger.info(f"문서 업로드 완료: doc_id={doc_id}, filename={filename}, chunks={len(chunks)}")
    
    return knowledge_doc


def ingest_documents_batch(
    db: Session,
    files: List[UploadFile],
    category: str,
    user_id: uuid.UUID
) -> List[BatchDocumentResult]:
    """
    여러 문서를 배치로 업로드 및 벡터 DB 저장
    """
    results = []
    
    logger.info(f"배치 업로드 시작: {len(files)}개 파일, 카테고리={category}")
    
    for file in files:
        filename = file.filename or "unknown.pdf"
        
        try:
            file.file.seek(0)
            
            knowledge_doc = ingest_document(
                db=db,
                file=file,
                category=category,
                user_id=user_id
            )
            
            result = BatchDocumentResult(
                success=True,
                filename=filename,
                document=DocumentResponse.model_validate(knowledge_doc),
                error=None
            )
            logger.info(f"배치 업로드 성공: {filename}")
            
        except HTTPException as e:
            result = BatchDocumentResult(
                success=False,
                filename=filename,
                document=None,
                error=str(e.detail)
            )
            logger.error(f"배치 업로드 실패: {filename}, 오류={e.detail}")
            
        except Exception as e:
            error_msg = str(e)
            result = BatchDocumentResult(
                success=False,
                filename=filename,
                document=None,
                error=f"문서 처리 중 오류가 발생했습니다: {error_msg}"
            )
            logger.error(f"배치 업로드 실패: {filename}, 오류={error_msg}", exc_info=True)
        
        results.append(result)
    
    success_count = sum(1 for r in results if r.success)
    failure_count = len(results) - success_count
    
    logger.info(f"배치 업로드 완료: 총 {len(results)}개, 성공 {success_count}개, 실패 {failure_count}개")
    
    return results


def get_documents(
    db: Session,
    category: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[KnowledgeDoc]:
    """
    문서 목록 조회
    """
    query = db.query(KnowledgeDoc)
    
    if category:
        query = query.filter(KnowledgeDoc.meta_info['category'].astext == category)
    
    documents = query.order_by(KnowledgeDoc.created_at.desc()).offset(offset).limit(limit).all()
    
    return documents


def delete_document(
    db: Session,
    doc_id: uuid.UUID
) -> None:
    """
    문서 삭제 (PostgreSQL + Milvus + S3)
    """
    knowledge_doc = db.query(KnowledgeDoc).filter(KnowledgeDoc.id == doc_id).first()
    
    if not knowledge_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="문서를 찾을 수 없습니다."
        )
    
    # S3에서 파일 삭제
    if knowledge_doc.storage_url:
        delete_from_s3(knowledge_doc.storage_url)
    
    if knowledge_doc.raw_pdf_url:
        delete_from_s3(knowledge_doc.raw_pdf_url)
    
    # Milvus에서 삭제
    try:
        collection = get_milvus_collection_safe()
        collection.delete(expr=f'doc_id == "{doc_id}"')
        collection.flush()
        
        logger.info(f"Milvus에서 문서 삭제 완료: doc_id={doc_id}")
        
    except Exception as e:
        logger.error(f"Milvus 삭제 실패: {str(e)}", exc_info=True)
    
    # PostgreSQL에서 삭제
    db.delete(knowledge_doc)
    db.commit()
    
    logger.info(f"문서 삭제 완료: doc_id={doc_id}")
