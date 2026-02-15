"""
관리자용 채팅 내역 조회 API
- 사용자 목록 (세션/메시지 통계 포함)
- 사용자별 세션 목록
- 세션별 메시지 내역
"""
from fastapi import APIRouter, Depends, Query, Path
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, or_
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import uuid

from app.core.database import get_db
from app.api.dependencies import require_admin
from app.models.user import User
from app.models.chat import ChatSession, ChatMessage

router = APIRouter()


# ── Response Schemas ──────────────────────────────────────────────

class ChatUserSummary(BaseModel):
    """사용자 요약 (채팅 통계 포함)"""
    user_id: uuid.UUID
    email: str
    nickname: str
    total_sessions: int
    total_messages: int
    last_chat_at: Optional[datetime] = None


class ChatUserListResponse(BaseModel):
    items: List[ChatUserSummary]
    total: int
    page: int
    size: int


class ChatSessionSummary(BaseModel):
    """세션 요약"""
    session_id: uuid.UUID
    title: Optional[str] = None
    started_at: datetime
    updated_at: datetime
    message_count: int


class ChatSessionListResponse(BaseModel):
    items: List[ChatSessionSummary]
    total: int


class ChatMessageDetail(BaseModel):
    """메시지 상세"""
    message_id: uuid.UUID
    role: str
    content: str
    is_emergency: bool
    is_retry: bool
    rag_sources: Optional[list] = None
    created_at: datetime


class ChatMessageListResponse(BaseModel):
    items: List[ChatMessageDetail]
    total: int
    session_title: Optional[str] = None
    user_nickname: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────

@router.get("/users", response_model=ChatUserListResponse)
def get_chat_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, description="이메일 또는 닉네임 검색"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    채팅 이력이 있는 사용자 목록 조회 (통계 포함)
    """
    # 서브쿼리: 사용자별 세션 수, 메시지 수, 마지막 대화 시각
    session_stats = (
        db.query(
            ChatSession.user_id,
            func.count(ChatSession.id).label("total_sessions"),
        )
        .group_by(ChatSession.user_id)
        .subquery()
    )

    message_stats = (
        db.query(
            ChatSession.user_id,
            func.count(ChatMessage.id).label("total_messages"),
            func.max(ChatMessage.created_at).label("last_chat_at"),
        )
        .join(ChatMessage, ChatMessage.session_id == ChatSession.id)
        .group_by(ChatSession.user_id)
        .subquery()
    )

    query = (
        db.query(
            User.id,
            User.email,
            User.nickname,
            func.coalesce(session_stats.c.total_sessions, 0).label("total_sessions"),
            func.coalesce(message_stats.c.total_messages, 0).label("total_messages"),
            message_stats.c.last_chat_at,
        )
        .outerjoin(session_stats, User.id == session_stats.c.user_id)
        .outerjoin(message_stats, User.id == message_stats.c.user_id)
        .filter(session_stats.c.total_sessions > 0)  # 세션이 있는 사용자만
    )

    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                User.email.ilike(search_term),
                User.nickname.ilike(search_term),
            )
        )

    total = query.count()
    rows = query.order_by(desc(message_stats.c.last_chat_at)).offset(skip).limit(limit).all()

    items = [
        ChatUserSummary(
            user_id=row.id,
            email=row.email,
            nickname=row.nickname,
            total_sessions=row.total_sessions,
            total_messages=row.total_messages,
            last_chat_at=row.last_chat_at,
        )
        for row in rows
    ]

    return ChatUserListResponse(
        items=items,
        total=total,
        page=(skip // limit) + 1,
        size=limit,
    )


@router.get("/users/{user_id}/sessions", response_model=ChatSessionListResponse)
def get_user_sessions(
    user_id: uuid.UUID = Path(..., description="사용자 ID"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    특정 사용자의 채팅 세션 목록 조회
    """
    message_count_sub = (
        db.query(
            ChatMessage.session_id,
            func.count(ChatMessage.id).label("msg_count"),
        )
        .group_by(ChatMessage.session_id)
        .subquery()
    )

    rows = (
        db.query(
            ChatSession.id,
            ChatSession.title,
            ChatSession.started_at,
            ChatSession.updated_at,
            func.coalesce(message_count_sub.c.msg_count, 0).label("message_count"),
        )
        .outerjoin(message_count_sub, ChatSession.id == message_count_sub.c.session_id)
        .filter(ChatSession.user_id == user_id)
        .order_by(desc(ChatSession.updated_at))
        .all()
    )

    items = [
        ChatSessionSummary(
            session_id=row.id,
            title=row.title,
            started_at=row.started_at,
            updated_at=row.updated_at,
            message_count=row.message_count,
        )
        for row in rows
    ]

    return ChatSessionListResponse(items=items, total=len(items))


@router.get("/sessions/{session_id}/messages", response_model=ChatMessageListResponse)
def get_session_messages(
    session_id: uuid.UUID = Path(..., description="세션 ID"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    특정 세션의 메시지 내역 조회 (시간순)
    """
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()

    user_nickname = None
    session_title = None
    if session:
        session_title = session.title
        user = db.query(User).filter(User.id == session.user_id).first()
        if user:
            user_nickname = user.nickname

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    items = [
        ChatMessageDetail(
            message_id=msg.id,
            role=msg.role,
            content=msg.content,
            is_emergency=msg.is_emergency,
            is_retry=msg.is_retry,
            rag_sources=msg.rag_sources,
            created_at=msg.created_at,
        )
        for msg in messages
    ]

    return ChatMessageListResponse(
        items=items,
        total=len(items),
        session_title=session_title,
        user_nickname=user_nickname,
    )
