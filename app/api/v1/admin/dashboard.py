"""
관리자 대시보드 API
"""
from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.models.user import User, UserRole
from app.models.chat import ChatSession
from app.models.knowledge import KnowledgeDoc
from pydantic import BaseModel

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")

class AdminStatsResponse(BaseModel):
    totalUsers: int
    totalSessions: int
    totalKnowledgeDocs: int

def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """관리자 권한 확인"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="관리자 권한이 필요합니다."
        )
    return current_user

@router.get(
    "/admin/stats",
    response_model=AdminStatsResponse,
    dependencies=[Depends(oauth2_scheme)]
)
async def get_dashboard_stats(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    관리자 대시보드 통계 조회
    - 전체 사용자 수
    - 전체 대화 세션 수
    - 전체 지식 베이스 문서 수
    """
    total_users = db.query(User).count()
    total_sessions = db.query(ChatSession).count()
    total_knowledge_docs = db.query(KnowledgeDoc).count()

    return AdminStatsResponse(
        totalUsers=total_users,
        totalSessions=total_sessions,
        totalKnowledgeDocs=total_knowledge_docs
    )

