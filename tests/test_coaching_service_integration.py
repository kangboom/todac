import json
import os
import uuid
from datetime import date, timedelta

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent.coaching.graph import build_coaching_graph
from app.models.baby import BabyProfile
from app.models.chat import ChatMessage, ChatSession
from app.models.coaching import ActionAttempt
from app.models.user import User
from app.services import chat_service, coaching_repository


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL이 필요합니다.")


async def _collect(generator):
    return [json.loads(item) async for item in generator]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sse_interaction_restore_and_request_idempotency(monkeypatch):
    engine = create_engine(TEST_DATABASE_URL)
    Session = sessionmaker(bind=engine)
    db = Session()
    graph = build_coaching_graph().compile(checkpointer=InMemorySaver())

    async def fake_graph():
        return graph

    monkeypatch.setattr(chat_service, "get_coaching_graph", fake_graph)

    user = User(
        email=f"coaching-{uuid.uuid4()}@example.com",
        password_hash="test",
        nickname="테스트 보호자",
    )
    db.add(user)
    db.flush()
    baby = BabyProfile(
        user_id=user.id,
        name="테스트 아기",
        birth_date=date.today() - timedelta(days=90),
        due_date=date.today() - timedelta(days=30),
        gender="F",
        birth_weight=1.8,
        medical_history=[],
    )
    db.add(baby)
    db.commit()

    first_request_id = uuid.uuid4()
    first = await _collect(chat_service.send_message_v2(
        db,
        user.id,
        baby.id,
        "수유 방법을 함께 계획하고 싶어요",
        request_id=first_request_id,
    ))
    interaction = next(event for event in first if event["type"] == "interaction")
    done = next(event for event in first if event["type"] == "done")
    session_id = uuid.UUID(done["session_id"])
    assert interaction["interaction"]["kind"] == "COACHING_CONSENT"
    assert done["coaching"]["status"] == "WAITING_USER"
    assert db.query(ChatMessage).filter(ChatMessage.session_id == session_id).count() == 2

    duplicate = await _collect(chat_service.send_message_v2(
        db,
        user.id,
        baby.id,
        "수유 방법을 함께 계획하고 싶어요",
        session_id=session_id,
        request_id=first_request_id,
    ))
    assert [event["type"] for event in duplicate] == ["interaction", "done"]
    assert duplicate[-1]["coaching"]["status"] == "WAITING_USER"
    assert db.query(ChatMessage).filter(ChatMessage.session_id == session_id).count() == 2

    stale = await _collect(chat_service.send_message_v2(
        db,
        user.id,
        baby.id,
        "코칭으로 진행",
        session_id=session_id,
        request_id=uuid.uuid4(),
        interaction_id=uuid.uuid4(),
        selected_option_id="accept",
    ))
    assert [event["type"] for event in stale] == ["interaction", "done"]
    assert db.query(ChatMessage).filter(ChatMessage.session_id == session_id).count() == 2

    resumed = await _collect(chat_service.send_message_v2(
        db,
        user.id,
        baby.id,
        "코칭으로 진행",
        session_id=session_id,
        request_id=uuid.uuid4(),
        interaction_id=uuid.UUID(interaction["interaction"]["id"]),
        selected_option_id="accept",
    ))
    resumed_interaction = next(event for event in resumed if event["type"] == "interaction")
    assert resumed_interaction["interaction"]["kind"] == "GOAL_INPUT"
    assert db.query(ChatMessage).filter(ChatMessage.session_id == session_id).count() == 4
    db.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_postgres_checkpointer_resumes_after_graph_rebuild():
    episode_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": episode_id}}
    initial_state = {
        "question": "수유 방법을 함께 계획하고 싶어요",
        "episode_id": episode_id,
        "phase": "CONSENT",
        "episode_status": "PENDING_CONSENT",
        "attempt_count": 0,
        "latest_resume": {"message": "수유 방법을 함께 계획하고 싶어요"},
        "resume_target": "mode_router",
        "goal_confirmed": False,
        "constraints": [],
        "action_options": [],
        "source_ids": [],
        "is_emergency": False,
    }

    async with AsyncPostgresSaver.from_conn_string(TEST_DATABASE_URL) as saver:
        await saver.setup()
        graph = build_coaching_graph().compile(checkpointer=saver)
        result = await graph.ainvoke(initial_state, config=config)
        assert result["pending_interaction"]["kind"] == "COACHING_CONSENT"
        interaction_id = result["pending_interaction"]["id"]

    config["configurable"]["business_context"] = {
        key: result.get(key)
        for key in coaching_repository.BUSINESS_STATE_KEYS
        if key in result
    }
    async with AsyncPostgresSaver.from_conn_string(TEST_DATABASE_URL) as saver:
        rebuilt_graph = build_coaching_graph().compile(checkpointer=saver)
        snapshot = await rebuilt_graph.aget_state(config)
        assert snapshot.next == ("wait_for_user",)
        assert "question" not in snapshot.values
        assert "goal" not in snapshot.values
        assert "response" not in snapshot.values
        result = await rebuilt_graph.ainvoke(Command(resume={
            "message": "코칭으로 진행",
            "request_id": str(uuid.uuid4()),
            "interaction_id": interaction_id,
            "selected_option_id": "accept",
        }), config=config)
        assert result["pending_interaction"]["kind"] == "GOAL_INPUT"


@pytest.mark.integration
def test_low_confidence_does_not_create_action_attempt():
    engine = create_engine(TEST_DATABASE_URL)
    Session = sessionmaker(bind=engine)
    db = Session()
    user = User(
        email=f"attempt-{uuid.uuid4()}@example.com",
        password_hash="test",
        nickname="테스트 보호자",
    )
    db.add(user)
    db.flush()
    baby = BabyProfile(
        user_id=user.id,
        name="테스트 아기",
        birth_date=date.today() - timedelta(days=90),
        due_date=date.today() - timedelta(days=30),
        gender="M",
        birth_weight=1.9,
        medical_history=[],
    )
    db.add(baby)
    db.flush()
    session = ChatSession(user_id=user.id, baby_id=baby.id)
    db.add(session)
    db.commit()
    episode = coaching_repository.create_episode(db, session.id)
    initial_version = episode.version
    episode = coaching_repository.claim_episode(db, episode.id, initial_version)
    assert episode.version == initial_version + 1
    with pytest.raises(coaching_repository.ConcurrentEpisodeUpdateError):
        coaching_repository.claim_episode(db, episode.id, initial_version)

    state = {
        "goal": "수유 상황 기록하기",
        "goal_confirmed": True,
        "success_criteria": "하루 세 번 기록",
        "time_horizon_days": 1,
        "selected_action": "수유량 기록",
        "action_plan": {"when": "수유 직후"},
        "confidence_score": 4,
        "attempt_count": 0,
        "episode_status": "WAITING_USER",
        "phase": "OPTIONS",
        "pending_interaction": {"id": str(uuid.uuid4()), "kind": "ACTION_SELECTION"},
    }
    episode = coaching_repository.save_episode_state(db, episode.id, episode.version, state, uuid.uuid4())
    assert db.query(ActionAttempt).filter(ActionAttempt.goal_id == episode.active_goal_id).count() == 0

    state.update({
        "confidence_score": 8,
        "attempt_count": 1,
        "phase": "CHECK_IN",
        "pending_interaction": {"id": str(uuid.uuid4()), "kind": "CHECK_IN"},
    })
    stale_version = episode.version
    coaching_repository.save_episode_state(db, episode.id, stale_version, state, uuid.uuid4())
    attempt = db.query(ActionAttempt).filter(ActionAttempt.goal_id == episode.active_goal_id).one()
    assert attempt.confidence_score == 8
    assert attempt.sequence == 1
    with pytest.raises(coaching_repository.ConcurrentEpisodeUpdateError):
        coaching_repository.save_episode_state(db, episode.id, stale_version, state, uuid.uuid4())
    db.close()
