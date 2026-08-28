import pytest

from app.agent.coaching import nodes


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
