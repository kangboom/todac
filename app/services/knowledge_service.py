"""
지식 베이스 서비스 (문서 업로드, 조회, 삭제)
"""
from sqlalchemy.orm import Session
from fastapi import HTTPException, status, UploadFile
from app.models.knowledge import KnowledgeDoc
from app.core.milvus_schema import MILVUS_COLLECTION_NAME, create_milvus_collection
from app.core.database import get_milvus_client
from app.dto.knowledge import BatchDocumentResult
from app.services.s3_service import upload_to_s3, delete_from_s3, generate_storage_paths
from app.services.parser_service import get_parser
from app.worker.tasks import process_document_task
from typing import List, Optional
import uuid
import hashlib
import logging

logger = logging.getLogger(__name__)


def _calculate_hash(content: bytes) -> str:
    """문서 내용의 SHA-256 해시값 계산"""
    return hashlib.sha256(content).hexdigest()


async def ingest_document_async(
    db: Session,
    file: UploadFile,
    category: str,
    user_id: uuid.UUID
) -> BatchDocumentResult:
    """
    문서 업로드 및 워커 태스크 실행 (비동기)
    1. 파일 해시 계산 (중복 체크)
    2. S3 Raw 업로드
    3. 워커 태스크 실행 (kick)
    """
    filename = file.filename
    doc_id = uuid.uuid4()
    
    try:
        # 1. 파일 읽기
        content = await file.read()
        file_size = len(content)
        
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="파일 내용이 비어있습니다."
            )
        
        # 2. 문서 해시 계산 (중복 업로드 체크)
        doc_hash = _calculate_hash(content)
        
        # 중복 문서 확인 (DB 조회)
        existing_doc = db.query(KnowledgeDoc).filter(KnowledgeDoc.doc_hash == doc_hash).first()
        if existing_doc:
            return BatchDocumentResult(
                success=False,
                filename=filename,
                document=None,
                error=f"이미 존재하는 문서입니다. (ID: {existing_doc.id})"
            )
        
        # 3. 파서 지원 여부 확인
        parser = get_parser(filename)
        if not parser:
            return BatchDocumentResult(
                success=False,
                filename=filename,
                document=None,
                error=f"지원하지 않는 파일 형식입니다."
            )

        # 4. S3 Raw 업로드
        storage_paths = generate_storage_paths(doc_id, filename)
        
        # upload_to_s3는 boto3 동기 함수이므로 실행
        # (파일이 매우 크면 여기서 블로킹 될 수 있지만, 일단 진행)
        upload_to_s3(
            content=content,
            s3_key=storage_paths.raw_pdf_key,
            content_type='application/pdf'
        )
        
        # 5. 워커 태스크 실행 (비동기)
        # kiq는 awaitable
        await process_document_task.kiq(
            doc_id_str=str(doc_id),
            raw_s3_key=storage_paths.raw_pdf_key,
            filename=filename,
            category=category,
            user_id_str=str(user_id),
            file_size=file_size,
            doc_hash=doc_hash
        )
        
        logger.info(f"문서 처리 태스크 예약 완료: {filename} (task_id={doc_id})")
        
        # 임시 응답 (DB에는 아직 없음)
        # 프론트엔드에서는 '처리 중' 상태로 표시할 수 있음
        return BatchDocumentResult(
            success=True,
            filename=filename,
            document=None, # 아직 생성 안됨
            error=None
        )

    except Exception as e:
        logger.error(f"문서 업로드 실패: {filename}, 오류: {e}")
        return BatchDocumentResult(
            success=False,
            filename=filename,
            document=None,
            error=str(e)
        )


async def ingest_documents_batch(
    db: Session,
    files: List[UploadFile],
    category: str,
    user_id: uuid.UUID
) -> List[BatchDocumentResult]:
    """
    여러 문서를 배치로 업로드 (비동기 워커 위임)
    """
    results = []
    
    logger.info(f"배치 업로드 시작: {len(files)}개 파일, 카테고리={category}")
    
    for file in files:
        result = await ingest_document_async(
            db=db,
            file=file,
            category=category,
            user_id=user_id
        )
        results.append(result)
    
    success_count = sum(1 for r in results if r.success)
    
    logger.info(f"배치 요청 처리 완료: 총 {len(results)}개, 성공(예약) {success_count}개")
    
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
        client = get_milvus_client()
        create_milvus_collection()  # 컬렉션이 없으면 생성
        
        client.delete(
            collection_name=MILVUS_COLLECTION_NAME,
            filter=f'doc_id == "{doc_id}"'
        )
        
        logger.info(f"Milvus에서 문서 삭제 완료: doc_id={doc_id}")
        
    except Exception as e:
        logger.error(f"Milvus 삭제 실패: {str(e)}", exc_info=True)
    
    # PostgreSQL에서 삭제
    db.delete(knowledge_doc)
    db.commit()
    
    logger.info(f"문서 삭제 완료: doc_id={doc_id}")
