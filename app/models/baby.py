"""
아기 정보 테이블 (BabyProfile)
"""
from sqlalchemy import Column, String, Date, Float, ForeignKey, DateTime, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.core.database import Base


class BabyProfile(Base):
    """아기 프로필 테이블 - 교정 연령 계산과 맞춤형 의료 상담을 위한 필수 데이터"""
    __tablename__ = "baby_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(50), nullable=False)
    birth_date = Column(Date, nullable=False, comment="실제 태어난 날 (예방접종 기준)")
    due_date = Column(Date, nullable=False, comment="출산 예정일 (교정 연령/발달 평가 기준)")
    gender = Column(String(10), nullable=True, comment="성별: M 또는 F")
    birth_weight = Column(Float, nullable=False, comment="출생 체중 (kg)")
    medical_history = Column(JSONB, default=list, nullable=False, comment="기저질환 리스트 (예: ['RDS', '황달'])")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # 제약조건
    __table_args__ = (
        CheckConstraint("gender IN ('M', 'F')", name="check_gender"),
    )

    # 관계 설정
    user = relationship("User", back_populates="baby_profiles")
    chat_sessions = relationship("ChatSession", back_populates="baby", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<BabyProfile(id={self.id}, name={self.name}, user_id={self.user_id})>"
