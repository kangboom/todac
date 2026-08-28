"""코칭 업무 상태 저장소.

LangGraph checkpoint와 별개로 조회·감사 가능한 업무 데이터를 멱등 저장한다.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from app.core.config import settings
from app.models.coaching import (
    ActionAttempt,
    ActionAttemptStatus,
    CoachingEpisode,
    CoachingEpisodeStatus,
    CoachingEvent,
    CoachingGoal,
    CoachingGoalStatus,
)

ACTIVE_STATUSES = (
    CoachingEpisodeStatus.PENDING_CONSENT.value,
    CoachingEpisodeStatus.ACTIVE.value,
    CoachingEpisodeStatus.WAITING_USER.value,
)


class ConcurrentEpisodeUpdateError(RuntimeError):
    pass


BUSINESS_STATE_KEYS = (
    "question",
    "previous_question",
    "response",
    "is_emergency",
    "mode",
    "goal",
    "success_criteria",
    "time_horizon_days",
    "goal_confirmed",
    "reality_summary",
    "constraints",
    "action_options",
    "selected_action",
    "action_plan",
    "confidence_score",
    "execution_result",
    "barrier",
    "review_route",
    "source_ids",
)


def _business_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {key: state.get(key) for key in BUSINESS_STATE_KEYS if key in state}


def get_graph_context(db: Session, episode_id: uuid.UUID) -> Dict[str, Any]:
    event = db.query(CoachingEvent).filter(
        CoachingEvent.episode_id == episode_id,
    ).order_by(CoachingEvent.created_at.desc()).first()
    return dict((event.payload or {}).get("state") or {}) if event else {}


def get_active_episode(db: Session, chat_session_id: uuid.UUID) -> Optional[CoachingEpisode]:
    return db.query(CoachingEpisode).options(joinedload(CoachingEpisode.active_goal)).filter(
        CoachingEpisode.chat_session_id == chat_session_id,
        CoachingEpisode.status.in_(ACTIVE_STATUSES),
    ).order_by(CoachingEpisode.created_at.desc()).first()


def get_latest_episode(db: Session, chat_session_id: uuid.UUID) -> Optional[CoachingEpisode]:
    return db.query(CoachingEpisode).options(joinedload(CoachingEpisode.active_goal)).filter(
        CoachingEpisode.chat_session_id == chat_session_id,
    ).order_by(CoachingEpisode.created_at.desc()).first()


def create_episode(db: Session, chat_session_id: uuid.UUID) -> CoachingEpisode:
    episode = CoachingEpisode(chat_session_id=chat_session_id)
    db.add(episode)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = get_active_episode(db, chat_session_id)
        if existing:
            return existing
        raise
    db.refresh(episode)
    return episode


def claim_episode(db: Session, episode_id: uuid.UUID, expected_version: int) -> CoachingEpisode:
    """그래프 실행 전에 Episode를 선점해 동일 체크포인트의 동시 재개를 막는다."""
    result = db.execute(
        update(CoachingEpisode)
        .where(CoachingEpisode.id == episode_id, CoachingEpisode.version == expected_version)
        .values(version=expected_version + 1, updated_at=datetime.now(timezone.utc))
    )
    if result.rowcount != 1:
        db.rollback()
        raise ConcurrentEpisodeUpdateError("다른 요청이 코칭 Episode를 먼저 실행하고 있습니다.")
    db.commit()
    return db.query(CoachingEpisode).options(joinedload(CoachingEpisode.active_goal)).filter(
        CoachingEpisode.id == episode_id
    ).one()


def _upsert_goal(db: Session, episode: CoachingEpisode, state: Dict[str, Any]) -> Optional[CoachingGoal]:
    if not state.get("goal_confirmed") or not state.get("goal"):
        return episode.active_goal

    goal = episode.active_goal
    if goal and goal.description != state["goal"]:
        goal.status = CoachingGoalStatus.CHANGED.value
        goal = None
    if not goal:
        goal = CoachingGoal(
            episode_id=episode.id,
            description=state["goal"],
            success_criteria=state.get("success_criteria") or "행동 실행 및 변화 기록",
            time_horizon_days=state.get("time_horizon_days", 1),
            confirmed_by_user=True,
            status=CoachingGoalStatus.ACTIVE.value,
            confirmed_at=datetime.now(timezone.utc),
        )
        db.add(goal)
        db.flush()
        episode.active_goal_id = goal.id
        episode.active_goal = goal
    else:
        goal.success_criteria = state.get("success_criteria") or goal.success_criteria
        goal.time_horizon_days = state.get("time_horizon_days", goal.time_horizon_days)
        goal.confirmed_by_user = True
        goal.status = CoachingGoalStatus.ACTIVE.value
    return goal


def _upsert_attempt(db: Session, goal: Optional[CoachingGoal], state: Dict[str, Any]) -> None:
    confidence_score = int(state.get("confidence_score") or 0)
    if (
        not goal
        or not state.get("selected_action")
        or int(state.get("attempt_count") or 0) < 1
        or confidence_score < settings.COACHING_CONFIDENCE_THRESHOLD
    ):
        return
    sequence = max(1, int(state.get("attempt_count", 1)))
    attempt = db.query(ActionAttempt).filter(
        ActionAttempt.goal_id == goal.id,
        ActionAttempt.sequence == sequence,
    ).first()
    if not attempt:
        attempt = ActionAttempt(
            goal_id=goal.id,
            sequence=sequence,
            selected_action=state["selected_action"],
            action_plan=state.get("action_plan") or {},
            confidence_score=confidence_score,
        )
        db.add(attempt)
    if state.get("execution_result"):
        attempt.result = state["execution_result"]
        attempt.barrier = state.get("barrier")
        attempt.status = ActionAttemptStatus.REPORTED.value
        attempt.reported_at = datetime.now(timezone.utc)


def save_episode_state(
    db: Session,
    episode_id: uuid.UUID,
    expected_version: int,
    state: Dict[str, Any],
    request_id: Optional[uuid.UUID],
    *,
    commit: bool = True,
) -> CoachingEpisode:
    episode = db.query(CoachingEpisode).filter(CoachingEpisode.id == episode_id).first()
    if not episode:
        raise LookupError("코칭 Episode를 찾을 수 없습니다.")

    goal = _upsert_goal(db, episode, state)
    _upsert_attempt(db, goal, state)

    status = state.get("episode_status", episode.status)
    phase = state.get("phase", episode.phase)
    completed_at = datetime.now(timezone.utc) if status in (
        CoachingEpisodeStatus.COMPLETED.value,
        CoachingEpisodeStatus.CANCELLED.value,
        CoachingEpisodeStatus.ESCALATED.value,
    ) else None

    result = db.execute(
        update(CoachingEpisode)
        .where(CoachingEpisode.id == episode_id, CoachingEpisode.version == expected_version)
        .values(
            status=status,
            phase=phase,
            attempt_count=state.get("attempt_count", episode.attempt_count),
            active_goal_id=episode.active_goal_id,
            pending_interaction=state.get("pending_interaction"),
            completed_at=completed_at,
            version=expected_version + 1,
            updated_at=datetime.now(timezone.utc),
        )
    )
    if result.rowcount != 1:
        db.rollback()
        raise ConcurrentEpisodeUpdateError("다른 요청이 코칭 상태를 먼저 변경했습니다.")

    if status == CoachingEpisodeStatus.COMPLETED.value and goal:
        goal.status = CoachingGoalStatus.ACHIEVED.value
        goal.completed_at = completed_at

    event_type = "INTERACTION_WAITING" if state.get("pending_interaction") else f"EPISODE_{status}"
    existing_event = None
    if request_id:
        existing_event = db.query(CoachingEvent).filter(
            CoachingEvent.episode_id == episode_id,
            CoachingEvent.request_id == request_id,
            CoachingEvent.event_type == event_type,
        ).first()
    if not existing_event:
        db.add(CoachingEvent(
            episode_id=episode_id,
            request_id=request_id,
            event_type=event_type,
            phase=phase,
            payload={
                "review_route": state.get("review_route"),
                "interaction_kind": (state.get("pending_interaction") or {}).get("kind"),
                "state": _business_state(state),
            },
        ))

    if commit:
        db.commit()
        return db.query(CoachingEpisode).filter(CoachingEpisode.id == episode_id).first()
    db.flush()
    db.refresh(episode)
    return episode
