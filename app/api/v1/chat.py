"""
채팅 관련 API
"""
from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import List, AsyncGenerator
import uuid
from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.models.user import User
from app.dto.chat import (
    ChatMessageRequest,
    ChatMessageSendResponse,
    ChatSessionResponse,
    ChatSessionDetailResponse,
    ChatMessageResponse
)
from app.services import chat_service
from app.services import chat_repository
from app.models.chat import ChatSession, ChatMessage

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


async def sse_generator(generator: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    """SSE 포맷으로 변환"""
    async for chunk in generator:
        yield f"data: {chunk}\n\n"


@router.post(
    "/chat/message",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(oauth2_scheme)]
)
async def send_message(
    request: ChatMessageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    메시지 전송 및 AI 응답 받기 (SSE 스트리밍)
    
    - **baby_id**: 아기 프로필 ID (필수)
    - **message**: 사용자 메시지 (필수)
    - **session_id**: 세션 ID (선택, 없으면 새로 생성)
    
    세션이 없으면 자동으로 새 세션을 생성하고, 있으면 기존 세션에 메시지를 추가합니다.
    응답은 Server-Sent Events (SSE) 형식으로 스트리밍됩니다.
    """
    generator = chat_service.send_message(
        db=db,
        user_id=current_user.id,
        baby_id=request.baby_id,
        question=request.message,
        session_id=request.session_id
    )
    
    return StreamingResponse(
        sse_generator(generator),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",  # Nginx 버퍼링 방지
            "Cache-Control": "no-cache", # 캐시 방지
            "Connection": "keep-alive"   # 연결 유지
        }
    )


@router.get(
    "/chat/sessions",
    response_model=List[ChatSessionResponse],
    dependencies=[Depends(oauth2_scheme)]
)
async def get_sessions(
    baby_id: uuid.UUID = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """내 채팅 세션 목록 조회 (baby_id로 필터링 가능)"""
    sessions = chat_repository.get_sessions(db, current_user.id, baby_id)
    
    # 메시지 개수 추가
    return [
        ChatSessionResponse(
            id=session.id,
            user_id=session.user_id,
            baby_id=session.baby_id,
            title=session.title,
            is_active=session.is_active,
            started_at=session.started_at,
            updated_at=session.updated_at,
            message_count=len(session.messages) if session.messages else 0
        )
        for session in sessions
    ]


@router.get(
    "/chat/sessions/{session_id}",
    response_model=ChatSessionDetailResponse,
    dependencies=[Depends(oauth2_scheme)]
)
async def get_session_detail(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """세션 상세 조회 (메시지 포함)"""
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="유효하지 않은 세션 ID입니다."
        )
    
    # 세션 존재 확인 및 권한 체크
    session = db.query(ChatSession).filter(
        ChatSession.id == session_uuid,
        ChatSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="세션을 찾을 수 없습니다."
        )
    
    # 메시지 조회
    messages = chat_repository.get_session_messages(db, session_uuid, current_user.id)
    
    # 응답 생성
    message_responses = [
        ChatMessageResponse(
            message_id=msg.id,
            session_id=msg.session_id,
            role=msg.role,
            content=msg.content,
            is_emergency=msg.is_emergency,
            rag_sources=msg.rag_sources,
            created_at=msg.created_at
        )
        for msg in messages
    ]
    
    return ChatSessionDetailResponse(
        id=session.id,
        user_id=session.user_id,
        baby_id=session.baby_id,
        title=session.title,
        is_active=session.is_active,
        started_at=session.started_at,
        updated_at=session.updated_at,
        message_count=len(messages),
        messages=message_responses
    )


@router.delete(
    "/chat/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(oauth2_scheme)]
)
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """채팅 세션 삭제"""
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="유효하지 않은 세션 ID입니다."
        )
    
    chat_repository.delete_session(db, session_uuid, current_user.id)
    return None
