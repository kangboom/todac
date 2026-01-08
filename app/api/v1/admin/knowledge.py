"""
관리자 지식 베이스 API
"""
from fastapi import APIRouter, Depends, status, UploadFile, File, Form, Query
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.models.user import User, UserRole
from fastapi import HTTPException
from app.dto.knowledge import (
    DocumentIngestRequest, 
    DocumentResponse, 
    DocumentListResponse,
    BatchDocumentResponse,
    BatchDocumentResult
)
from app.services.knowledge_service import knowledge_service

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """관리자 권한 확인"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="관리자 권한이 필요합니다."
        )
    return current_user


@router.post(
    "/admin/knowledge/ingest",
    response_model=BatchDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(oauth2_scheme)]
)
async def ingest_documents(
    files: List[UploadFile] = File(..., description="업로드할 문서 파일들 (PDF, 1개 이상 선택 가능)"),
    category: str = Form(..., description="문서 카테고리 (모든 파일에 동일하게 적용, 예: 식이, 수면, 호흡, 발달, 예방접종, 피부, 응급)"),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    문서 업로드 및 벡터 DB 저장 (단일 또는 배치)
    
    - **files**: 업로드할 문서 파일들 (PDF, 1개 이상 선택 가능)
    - **category**: 문서 카테고리 (모든 파일에 동일하게 적용)
    
    각 문서가 업로드되면:
    1. 텍스트 추출 및 청킹
    2. OpenAI 임베딩 생성
    3. Milvus 벡터 DB 저장
    4. PostgreSQL 메타데이터 저장
    
    반환값:
    - **results**: 각 파일의 업로드 결과 (성공/실패 포함)
    - **total**: 전체 파일 수
    - **success_count**: 성공한 파일 수
    - **failure_count**: 실패한 파일 수
    """
    # 카테고리 유효성 검사
    valid_categories = ["식이", "수면", "호흡", "발달", "예방접종", "피부", "응급", "기타"]
    if category not in valid_categories:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"유효하지 않은 카테고리입니다. 가능한 값: {', '.join(valid_categories)}"
        )
    
    # 파일 개수 확인
    if not files or len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="업로드할 파일이 없습니다."
        )
    
    if len(files) > 50:  # 최대 50개까지 제한
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="한 번에 최대 50개까지 업로드할 수 있습니다."
        )
    
    # 배치 업로드 실행 (단일 파일이어도 동일한 방식으로 처리)
    results = knowledge_service.ingest_documents_batch(
        db=db,
        files=files,
        category=category,
        user_id=current_user.id
    )
    
    # 결과 변환
    batch_results = []
    for result in results:
        batch_result = BatchDocumentResult(
            success=result["success"],
            filename=result["filename"],
            document=DocumentResponse.model_validate(result["document"]) if result["document"] else None,
            error=result["error"]
        )
        batch_results.append(batch_result)
    
    success_count = sum(1 for r in batch_results if r.success)
    failure_count = len(batch_results) - success_count
    
    return BatchDocumentResponse(
        results=batch_results,
        total=len(batch_results),
        success_count=success_count,
        failure_count=failure_count
    )


@router.get(
    "/admin/knowledge/docs",
    response_model=DocumentListResponse,
    dependencies=[Depends(oauth2_scheme)]
)
async def get_documents(
    category: Optional[str] = Query(None, description="카테고리 필터"),
    limit: int = Query(100, ge=1, le=1000, description="최대 개수"),
    offset: int = Query(0, ge=0, description="오프셋"),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    문서 목록 조회
    
    - **category**: 카테고리 필터 (선택)
    - **limit**: 최대 개수 (1-1000)
    - **offset**: 오프셋
    """
    documents = knowledge_service.get_documents(
        db=db,
        category=category,
        limit=limit,
        offset=offset
    )
    
    # 전체 개수 조회 (필터 적용)
    from app.models.knowledge import KnowledgeDoc
    total_query = db.query(KnowledgeDoc)
    if category:
        # meta_info JSONB 필드에서 category 필터링
        total_query = total_query.filter(KnowledgeDoc.meta_info['category'].astext == category)
    total = total_query.count()
    
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(doc) for doc in documents],
        total=total,
        limit=limit,
        offset=offset
    )


@router.delete(
    "/admin/knowledge/docs/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(oauth2_scheme)]
)
async def delete_document(
    doc_id: str,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    문서 삭제
    
    - **doc_id**: 문서 ID
    
    문서가 삭제되면:
    1. Milvus에서 해당 문서의 모든 청크 삭제
    2. PostgreSQL에서 메타데이터 삭제
    """
    try:
        doc_uuid = uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="유효하지 않은 문서 ID입니다."
        )
    
    knowledge_service.delete_document(db=db, doc_id=doc_uuid)
    return None

