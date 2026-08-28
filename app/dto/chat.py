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
    request_id: Optional[uuid.UUID] = Field(None, description="클라이언트 요청 멱등성 ID")
    interaction_id: Optional[uuid.UUID] = Field(None, description="현재 대기 중인 코칭 상호작용 ID")
    selected_option_id: Optional[str] = Field(None, description="코칭 선택지 ID")


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
    messages: List[ChatMessageResponse] = Field(default_factory=list)
    coaching: Optional[Dict[str, Any]] = None


class ChatMessageSendResponse(BaseModel):
    """메시지 전송 응답"""
    response: str = Field(..., description="AI 응답")
    session_id: str = Field(..., description="세션 ID")
    is_emergency: bool = Field(..., description="응급 상황 여부")
    rag_sources: Optional[List[Dict[str, Any]]] = Field(None, description="참조 문서 정보")
    qna_sources: Optional[List[Dict[str, Any]]] = Field(None, description="QnA 참조 정보")
    response_time: float = Field(..., description="응답 시간 (초)")
    coaching: Optional[Dict[str, Any]] = None


class ConversationMessage(BaseModel):
    """대화 이력 메시지 (에이전트용)"""
    role: str = Field(..., description="메시지 역할 (user/assistant)")
    content: str = Field(..., description="메시지 내용")
    is_retry: bool = Field(..., description="재질문 모드 여부")
