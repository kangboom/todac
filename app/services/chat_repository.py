"""
채팅 관련 데이터베이스 접근 레이어 (Repository)
"""
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.models.chat import ChatSession, ChatMessage, MessageRole
from app.dto.chat import ConversationMessage
from typing import List
import uuid
import logging

logger = logging.getLogger(__name__)


def get_or_create_session(
    db: Session,
    user_id: uuid.UUID,
    baby_id: uuid.UUID,
    session_id: uuid.UUID = None
) -> ChatSession:
    """세션 가져오기 또는 생성"""
    if session_id:
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id
        ).first()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="세션을 찾을 수 없습니다."
            )
        
        return session
    else:
        # 새 세션 생성
        new_session = ChatSession(
            user_id=user_id,
            baby_id=baby_id
        )
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        return new_session


def get_conversation_history(
    db: Session,
    session_id: uuid.UUID,
    limit: int = 10
) -> List[ConversationMessage]:
    """대화 이력 가져오기"""
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.created_at.desc()).limit(limit).all()
    
    # 최신순으로 정렬 (오래된 것부터)
    messages.reverse()
    
    return [
        ConversationMessage(
            role="user" if msg.role == MessageRole.USER.value else "assistant",
            content=msg.content
        )
        for msg in messages
    ]


def get_sessions(db: Session, user_id: uuid.UUID, baby_id: uuid.UUID = None) -> List[ChatSession]:
    """사용자의 세션 조회 (baby_id로 필터링 가능)"""
    query = db.query(ChatSession).filter(
        ChatSession.user_id == user_id
    )
    
    if baby_id:
        query = query.filter(ChatSession.baby_id == baby_id)
    
    sessions = query.order_by(ChatSession.updated_at.desc()).all()
    
    return sessions


def get_session_messages(
    db: Session,
    session_id: uuid.UUID,
    user_id: uuid.UUID
) -> List[ChatMessage]:
    """세션의 모든 메시지 조회"""
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user_id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="세션을 찾을 수 없습니다."
        )
    
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.created_at.asc()).all()
    
    return messages


def delete_session(
    db: Session,
    session_id: uuid.UUID,
    user_id: uuid.UUID
) -> None:
    """세션 삭제 (소속된 메시지도 함께 삭제됨)"""
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user_id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="세션을 찾을 수 없습니다."
        )
        
    db.delete(session)
    db.commit()
