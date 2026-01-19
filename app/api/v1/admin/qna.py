from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.api.dependencies import require_admin
from app.dto.qna import QnACreateRequest, QnAResponse
from app.services import qna_service
from app.models.user import User
from app.models.qna import OfficialQnA

router = APIRouter()

@router.post("/", response_model=QnAResponse)
def create_qna(
    request: QnACreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """
    공식 QnA 데이터 등록
    - DB에 저장
    - Milvus에 임베딩하여 저장 (검색용)
    """
    try:
        qna = qna_service.ingest_qna(
            db=db,
            question=request.question,
            answer=request.answer,
            source=request.source,
            category=request.category
        )
        return qna
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"QnA 등록 실패: {str(e)}"
        )

@router.get("/", response_model=List[QnAResponse])
def get_qna_list(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """
    QnA 목록 조회 (최신순)
    """
    qnas = db.query(OfficialQnA).order_by(OfficialQnA.id.desc()).offset(skip).limit(limit).all()
    return qnas

@router.post("/sync")
def sync_qna_db(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """
    DB의 모든 QnA 데이터를 Milvus로 동기화 (재색인)
    - 기존 Milvus 컬렉션을 삭제하고 재생성합니다.
    - 데이터 양에 따라 시간이 소요될 수 있습니다.
    """
    try:
        count = qna_service.sync_all_qna_to_milvus(db)
        return {"message": f"총 {count}개의 QnA 데이터가 동기화되었습니다.", "count": count}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"동기화 실패: {str(e)}"
        )
