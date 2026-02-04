from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List

from app.core.database import get_db
from app.api.dependencies import require_admin
from app.dto.feedback import FeedbackListResponse, AdminFeedbackResponse
from app.models.user import User
from app.models.feedback import Feedback
from app.models.chat import ChatMessage

router = APIRouter()

@router.get("/", response_model=FeedbackListResponse)
def get_feedback_list(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """
    사용자 피드백 목록 조회 (관리자용)
    - 최신순 정렬
    - 질문/답변 내용 포함
    """
    total = db.query(Feedback).count()
    feedbacks = db.query(Feedback).order_by(desc(Feedback.created_at)).offset(skip).limit(limit).all()
    
    items = []
    for fb in feedbacks:
        # 피드백 대상 메시지 (답변)
        answer_msg = fb.message
        
        # 세션 정보 및 사용자 정보 접근
        # (ChatMessage -> ChatSession -> User)
        session = answer_msg.session
        user = session.user if session else None
        
        # 해당 답변의 직전 질문 찾기
        # 동일 세션 내에서, 답변 메시지보다 이전에 생성된 메시지 중 가장 최신 것 (USER 역할)
        question_msg = db.query(ChatMessage).filter(
            ChatMessage.session_id == answer_msg.session_id,
            ChatMessage.role == 'USER',
            ChatMessage.created_at < answer_msg.created_at
        ).order_by(desc(ChatMessage.created_at)).first()
        
        items.append(AdminFeedbackResponse(
            id=fb.id,
            score=fb.score,
            comment=fb.comment,
            created_at=fb.created_at,
            message_id=fb.message_id,
            answer=answer_msg.content,
            question=question_msg.content if question_msg else None,
            session_id=answer_msg.session_id,
            user_email=user.email if user else None,
            user_nickname=user.nickname if user else None
        ))
    
    return FeedbackListResponse(
        items=items,
        total=total,
        page=(skip // limit) + 1,
        size=limit
    )
