"""
답변 평가 테이블 (Feedback)
"""
from sqlalchemy import Column, Text, Integer, ForeignKey, DateTime, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.core.database import Base


class Feedback(Base):
    """피드백 테이블 - RAG 품질 개선을 위한 사용자 피드백"""
    __tablename__ = "feedbacks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    message_id = Column(UUID(as_uuid=True), ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False, index=True)
    score = Column(Integer, nullable=False, comment="만족도 점수 (1~5)")
    comment = Column(Text, nullable=True, comment="개선 의견")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # 제약조건
    __table_args__ = (
        CheckConstraint("score >= 1 AND score <= 5", name="check_feedback_score"),
    )

    # 관계 설정
    message = relationship("ChatMessage", back_populates="feedbacks")

    def __repr__(self):
        return f"<Feedback(id={self.id}, message_id={self.message_id}, score={self.score})>"

