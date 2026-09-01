import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agent.v2.graph import build_coaching_graph
from app.services.coaching_repository import BUSINESS_STATE_KEYS


def initial_state(question: str = "아기 수유 방법을 함께 계획하고 싶어요"):
    episode_id = str(uuid.uuid4())
    return episode_id, {
        "question": question,
        "previous_question": question,
        "episode_id": episode_id,
        "phase": "CONSENT",
        "episode_status": "PENDING_CONSENT",
        "attempt_count": 0,
        "latest_resume": {"message": question},
        "resume_target": "mode_router",
        "goal_confirmed": False,
        "constraints": [],
        "action_options": [],
        "source_ids": [],
        "is_emergency": False,
    }


async def resume(graph, config, message, selected=None):
    result = await graph.ainvoke(Command(resume={
        "message": message,
        "request_id": str(uuid.uuid4()),
        "interaction_id": "test-interaction",
        "selected_option_id": selected,
    }), config=config)
    config["configurable"]["business_context"] = {
        key: result.get(key) for key in BUSINESS_STATE_KEYS if key in result
    }
    return result


def remember_context(config, result):
    config["configurable"]["business_context"] = {
        key: result.get(key) for key in BUSINESS_STATE_KEYS if key in result
    }


@pytest.mark.asyncio
async def test_complete_grow_coaching_loop():
    episode_id, state = initial_state()
    graph = build_coaching_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": episode_id}}

    result = await graph.ainvoke(state, config=config)
    assert result["pending_interaction"]["kind"] == "COACHING_CONSENT"
    remember_context(config, result)

    result = await resume(graph, config, "코칭으로 진행", "accept")
    assert result["pending_interaction"]["kind"] == "GOAL_INPUT"

    result = await resume(graph, config, "수유할 때 덜 힘들어하도록 돕고 싶어요")
    assert result["pending_interaction"]["kind"] == "GOAL_CONFIRMATION"

    result = await resume(graph, config, "이 목표로 진행", "confirm")
    assert result["pending_interaction"]["kind"] == "REALITY_INPUT"

    result = await resume(graph, config, "40ml 정도 먹은 뒤 몸을 젖히고 중간 트림은 시도했어요")
    assert result["pending_interaction"]["kind"] == "REALITY_CONFIRMATION"

    result = await resume(graph, config, "맞아요", "confirm")
    assert result["pending_interaction"]["kind"] == "ACTION_SELECTION"

    result = await resume(graph, config, "상황과 반응을 기록해 보기", "option-1")
    assert result["pending_interaction"]["kind"] == "PLAN_CONFIRMATION"

    result = await resume(graph, config, "이 계획으로 진행", "confirm")
    assert result["pending_interaction"]["kind"] == "CHECK_IN"
    assert result["attempt_count"] == 1

    result = await resume(graph, config, "성공 기준을 달성했고 전보다 좋아졌어요")
    assert result["pending_interaction"]["kind"] == "COMPLETION_CONFIRMATION"

    result = await resume(graph, config, "목표 달성", "confirm")
    assert result["episode_status"] == "COMPLETED"
    assert result["phase"] == "COMPLETE"


@pytest.mark.asyncio
async def test_revising_plan_returns_to_options_without_creating_attempt():
    episode_id, state = initial_state()
    graph = build_coaching_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": episode_id}}
    result = await graph.ainvoke(state, config=config)
    remember_context(config, result)
    await resume(graph, config, "코칭으로 진행", "accept")
    await resume(graph, config, "수유 상황을 기록하고 싶어요")
    await resume(graph, config, "진행", "confirm")
    await resume(graph, config, "먹은 양과 시간을 기록 중이에요")
    await resume(graph, config, "맞아요", "confirm")
    await resume(graph, config, "상황과 반응을 기록해 보기", "option-1")
    result = await resume(graph, config, "다른 방법 선택", "revise")
    assert result["pending_interaction"]["kind"] == "ACTION_SELECTION"
    assert result["attempt_count"] == 0


@pytest.mark.asyncio
async def test_declining_coaching_returns_general_rag_answer():
    episode_id, state = initial_state()
    graph = build_coaching_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": episode_id}}
    result = await graph.ainvoke(state, config=config)
    remember_context(config, result)

    result = await resume(graph, config, "정보만 보기", "decline")
    assert result["episode_status"] == "CANCELLED"
    assert result["phase"] == "COMPLETE"
    assert result["source_ids"] == ["test-source"]


@pytest.mark.asyncio
async def test_emergency_input_interrupts_any_waiting_phase():
    episode_id, state = initial_state()
    graph = build_coaching_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": episode_id}}
    result = await graph.ainvoke(state, config=config)
    remember_context(config, result)

    result = await resume(graph, config, "아기가 지금 숨을 못 쉬고 파래요", "accept")
    assert result["episode_status"] == "ESCALATED"
    assert result["is_emergency"] is True
    assert "119" in result["response"]


@pytest.mark.asyncio
async def test_new_goal_action_starts_at_goal_without_repeating_consent():
    episode_id, state = initial_state("새 목표 시작")
    state["resume_target"] = "goal_prepare"
    graph = build_coaching_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": episode_id}}

    result = await graph.ainvoke(state, config=config)
    assert result["pending_interaction"]["kind"] == "GOAL_INPUT"
    assert result["phase"] == "GOAL"
