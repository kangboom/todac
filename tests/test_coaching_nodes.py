import pytest

from app.agent.v2 import nodes
from app.agent.v2.nodes import common


@pytest.mark.asyncio
async def test_apply_goal_node_revises_existing_goal_with_feedback(monkeypatch):
    captured = {}

    async def fake_structured_output(prompt, fallback):
        captured["prompt"] = prompt
        captured["fallback"] = fallback
        return {
            "goal": "하루 한 번 수유 기록하기",
            "success_criteria": "하루 한 번 기록을 남긴다",
            "time_horizon_days": 2,
        }

    monkeypatch.setattr(common, "structured_output", fake_structured_output)

    result = await nodes.apply_goal_node(
        {
            "question": "수유 상황을 개선하고 싶어요",
            "goal": "하루 세 번 수유 기록하기",
            "success_criteria": "하루 세 번 기록을 남긴다",
            "time_horizon_days": 1,
            "latest_resume": {
                "message": "기록 횟수를 하루 한 번으로 줄이고 싶어요"
            },
        }
    )

    assert "현재 목표: 하루 세 번 수유 기록하기" in captured["prompt"]
    assert (
        "보호자 수정 의견: 기록 횟수를 하루 한 번으로 줄이고 싶어요"
        in captured["prompt"]
    )
    assert result == {
        "goal": "하루 한 번 수유 기록하기",
        "success_criteria": "하루 한 번 기록을 남긴다",
        "time_horizon_days": 2,
        "goal_confirmed": False,
    }


@pytest.mark.asyncio
async def test_review_routes_no_progress_to_change_option():
    command = await nodes.review_node({
        "goal": "수유 행동 기록하기",
        "success_criteria": "하루 세 번 기록",
        "selected_action": "수유량 기록",
        "execution_result": "실행하기 너무 어려워서 하지 못했어요",
        "attempt_count": 1,
    })
    assert command.goto == "options_prepare"


@pytest.mark.asyncio
async def test_review_escalates_after_max_adjustments():
    command = await nodes.review_node({
        "goal": "수유 행동 기록하기",
        "success_criteria": "하루 세 번 기록",
        "selected_action": "수유량 기록",
        "execution_result": "여전히 실행하지 못했어요",
        "attempt_count": 3,
    })
    assert command.goto == "escalated_response"
