"""GROW 코칭의 업무 상태를 저장하는 모델."""
from __future__ import annotations

import enum
import uuid

from sqlalchemy import (
    CheckConstraint,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class CoachingEpisodeStatus(str, enum.Enum):
    PENDING_CONSENT = "PENDING_CONSENT"
    ACTIVE = "ACTIVE"
    WAITING_USER = "WAITING_USER"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    ESCALATED = "ESCALATED"


class CoachingPhase(str, enum.Enum):
    CONSENT = "CONSENT"
    GOAL = "GOAL"
    REALITY = "REALITY"
    OPTIONS = "OPTIONS"
    WILL = "WILL"
    CHECK_IN = "CHECK_IN"
    REVIEW = "REVIEW"
    COMPLETE = "COMPLETE"


class CoachingGoalStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    ACHIEVED = "ACHIEVED"
    CHANGED = "CHANGED"
    CANCELLED = "CANCELLED"


class ActionAttemptStatus(str, enum.Enum):
    PLANNED = "PLANNED"
    REPORTED = "REPORTED"
    ABANDONED = "ABANDONED"


class CoachingEpisode(Base):
    __tablename__ = "coaching_episodes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status = Column(String(30), nullable=False, default=CoachingEpisodeStatus.PENDING_CONSENT.value)
    phase = Column(String(20), nullable=False, default=CoachingPhase.CONSENT.value)
    attempt_count = Column(Integer, nullable=False, default=0)
    version = Column(Integer, nullable=False, default=1)
    active_goal_id = Column(
        UUID(as_uuid=True),
        ForeignKey("coaching_goals.id", name="fk_episode_active_goal", use_alter=True, ondelete="SET NULL"),
        nullable=True,
    )
    pending_interaction = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    chat_session = relationship("ChatSession", back_populates="coaching_episodes")
    goals = relationship(
        "CoachingGoal",
        back_populates="episode",
        cascade="all, delete-orphan",
        foreign_keys="CoachingGoal.episode_id",
    )
    active_goal = relationship("CoachingGoal", foreign_keys=[active_goal_id], post_update=True)
    events = relationship("CoachingEvent", back_populates="episode", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("attempt_count >= 0", name="check_episode_attempt_count"),
        CheckConstraint("version >= 1", name="check_episode_version"),
        Index(
            "uq_active_episode_per_chat_session",
            "chat_session_id",
            unique=True,
            postgresql_where=text("status IN ('PENDING_CONSENT', 'ACTIVE', 'WAITING_USER')"),
        ),
    )


class CoachingGoal(Base):
    __tablename__ = "coaching_goals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    episode_id = Column(
        UUID(as_uuid=True),
        ForeignKey("coaching_episodes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    description = Column(Text, nullable=False)
    success_criteria = Column(Text, nullable=False)
    time_horizon_days = Column(Integer, nullable=False, default=1)
    confirmed_by_user = Column(Boolean, nullable=False, default=False)
    status = Column(String(20), nullable=False, default=CoachingGoalStatus.DRAFT.value)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    confirmed_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    episode = relationship("CoachingEpisode", back_populates="goals", foreign_keys=[episode_id])
    attempts = relationship("ActionAttempt", back_populates="goal", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("time_horizon_days >= 1", name="check_goal_time_horizon"),
    )


class ActionAttempt(Base):
    __tablename__ = "coaching_action_attempts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    goal_id = Column(UUID(as_uuid=True), ForeignKey("coaching_goals.id", ondelete="CASCADE"), nullable=False, index=True)
    sequence = Column(Integer, nullable=False)
    selected_action = Column(Text, nullable=False)
    action_plan = Column(JSONB, nullable=False, default=dict)
    confidence_score = Column(Integer, nullable=False)
    result = Column(Text, nullable=True)
    barrier = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default=ActionAttemptStatus.PLANNED.value)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    reported_at = Column(DateTime(timezone=True), nullable=True)

    goal = relationship("CoachingGoal", back_populates="attempts")

    __table_args__ = (
        UniqueConstraint("goal_id", "sequence", name="uq_action_attempt_sequence"),
        CheckConstraint("sequence >= 1", name="check_action_attempt_sequence"),
        CheckConstraint("confidence_score BETWEEN 0 AND 10", name="check_action_confidence"),
    )


class CoachingEvent(Base):
    __tablename__ = "coaching_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    episode_id = Column(UUID(as_uuid=True), ForeignKey("coaching_episodes.id", ondelete="CASCADE"), nullable=False, index=True)
    request_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    event_type = Column(String(50), nullable=False)
    phase = Column(String(20), nullable=False)
    payload = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    episode = relationship("CoachingEpisode", back_populates="events")

    __table_args__ = (
        UniqueConstraint("episode_id", "request_id", "event_type", name="uq_coaching_event_request_type"),
    )
