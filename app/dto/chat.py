"""
채팅 메시지 요청/응답 양식
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid


class ChatMessageRequest(BaseModel):
    """메시지 전송 요청"""
    baby_id: uuid.UUID = Field(..., description="아기 프로필 ID")
    message: str = Field(..., min_length=1, description="사용자 메시지")
    session_id: Optional[uuid.UUID] = Field(None, description="세션 ID (없으면 새로 생성)")


class CreateSessionRequest(BaseModel):
    """세션 생성 요청"""
    baby_id: uuid.UUID = Field(..., description="아기 프로필 ID")


class ChatMessageResponse(BaseModel):
    """메시지 응답"""
    message_id: uuid.UUID
    session_id: uuid.UUID
    role: str  # "USER" or "ASSISTANT"
    content: str
    is_emergency: bool
    rag_sources: Optional[List[Dict[str, Any]]] = None
    qna_sources: Optional[List[Dict[str, Any]]] = None
    created_at: datetime


class ChatSessionResponse(BaseModel):
    """세션 응답"""
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    user_id: uuid.UUID
    baby_id: uuid.UUID
    title: Optional[str]
    is_active: bool
    started_at: datetime
    updated_at: datetime
    message_count: Optional[int] = 0  # 메시지 개수 (선택)


class ChatSessionDetailResponse(ChatSessionResponse):
    """세션 상세 응답 (메시지 포함)"""
    messages: List[ChatMessageResponse] = []


class ChatMessageSendResponse(BaseModel):
    """메시지 전송 응답"""
    response: str = Field(..., description="AI 응답")
    session_id: str = Field(..., description="세션 ID")
    is_emergency: bool = Field(..., description="응급 상황 여부")
    rag_sources: Optional[List[Dict[str, Any]]] = Field(None, description="참조 문서 정보")
    qna_sources: Optional[List[Dict[str, Any]]] = Field(None, description="QnA 참조 정보")
    response_time: float = Field(..., description="응답 시간 (초)")


class ConversationMessage(BaseModel):
    """대화 이력 메시지 (에이전트용)"""
    role: str = Field(..., description="메시지 역할 (user/assistant)")
    content: str = Field(..., description="메시지 내용")