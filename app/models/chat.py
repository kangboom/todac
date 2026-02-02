"""
채팅 관련 테이블 (ChatSession, ChatMessage)
"""
from sqlalchemy import Column, String, Text, Boolean, ForeignKey, DateTime, Integer, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum

from app.core.database import Base


class MessageRole(str, enum.Enum):
    """메시지 역할"""
    USER = "USER"
    ASSISTANT = "ASSISTANT"


class ChatSession(Base):
    """채팅방/세션 테이블 - 대화의 맥락을 유지하는 단위"""
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    baby_id = Column(UUID(as_uuid=True), ForeignKey("baby_profiles.id", ondelete="CASCADE"), nullable=False, index=True, comment="상담 대상 아기 ID (Context 주입용)")
    title = Column(String(100), nullable=True, comment="세션 제목 (첫 질문으로 자동 생성)")
    is_active = Column(Boolean, default=True, nullable=False)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # [추가] 대화 상태 저장 (Context 유지용)
    missing_info = Column(JSONB, nullable=True, comment="부족한 정보 목록 (예: ['아기 월령', '수유량']) - 다음 턴에서 참조")

    # 관계 설정
    user = relationship("User", back_populates="chat_sessions")
    baby = relationship("BabyProfile", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at")

    def __repr__(self):
        return f"<ChatSession(id={self.id}, user_id={self.user_id}, baby_id={self.baby_id}, title={self.title})>"


class ChatMessage(Base):
    """메시지 상세 테이블 - 실제 오고 간 대화 내용과 AI의 판단 근거"""
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False, comment="화자: USER 또는 ASSISTANT")
    content = Column(Text, nullable=False, comment="대화 내용 텍스트")
    is_retry = Column(Boolean, default=False, nullable=False, comment="재질문 모드 여부")
    is_emergency = Column(Boolean, default=False, nullable=False, index=True, comment="응급 상황 감지 여부 (통계 분석용)")
    rag_sources = Column(JSONB, nullable=True, comment="참조 문서 정보 (예: [{'doc_id': '...', 'score': 0.9}])")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    # 제약조건
    __table_args__ = (
        CheckConstraint("role IN ('USER', 'ASSISTANT')", name="check_message_role"),
    )

    # 관계 설정
    session = relationship("ChatSession", back_populates="messages")
    feedbacks = relationship("Feedback", back_populates="message", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, session_id={self.session_id}, role={self.role}, is_emergency={self.is_emergency})>"
