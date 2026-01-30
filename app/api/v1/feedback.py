"""
피드백 API
"""
from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import List
import uuid
from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.models.user import User
from app.dto.feedback import FeedbackCreateRequest, FeedbackResponse
from app.models.feedback import Feedback
from app.models.chat import ChatMessage, ChatSession

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(oauth2_scheme)]
)
async def create_feedback(
    request: FeedbackCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    메시지에 대한 피드백 생성
    - 사용자 본인의 메시지(에 대한 답변)에만 피드백 가능
    """
    # 1. 메시지 존재 여부 및 권한 확인
    # 메시지를 조회하면서 세션 정보도 함께 가져와서 본인의 세션인지 확인
    message = db.query(ChatMessage).join(ChatSession).filter(
        ChatMessage.id == request.message_id,
        ChatSession.user_id == current_user.id
    ).first()

    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="메시지를 찾을 수 없거나 권한이 없습니다."
        )
    
    # 2. ASSISTANT 메시지에만 피드백 가능하도록 제한 (선택사항)
    if message.role != "ASSISTANT":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="AI의 답변에만 피드백을 남길 수 있습니다."
        )

    # 3. 기존 피드백 확인 (중복 방지 또는 업데이트)
    # 한 메시지에 하나의 피드백만 가능하다고 가정 (업데이트 로직으로 구현 가능하나 여기선 생성만)
    existing_feedback = db.query(Feedback).filter(
        Feedback.message_id == request.message_id
    ).first()

    if existing_feedback:
        # 이미 있으면 업데이트
        existing_feedback.score = request.score
        existing_feedback.comment = request.comment
        db.commit()
        db.refresh(existing_feedback)
        return existing_feedback

    # 4. 새 피드백 생성
    new_feedback = Feedback(
        message_id=request.message_id,
        score=request.score,
        comment=request.comment
    )
    
    db.add(new_feedback)
    db.commit()
    db.refresh(new_feedback)
    
    return new_feedback
