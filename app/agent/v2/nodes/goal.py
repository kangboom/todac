"""GROW의 목표 설정과 목표 확인 노드."""
from __future__ import annotations

from typing import Any, Dict

from langgraph.types import Command

from app.agent.v2.nodes import common
from app.agent.v2.prompts import GOAL_PROMPT, GOAL_REVISION_PROMPT
from app.agent.v2.state import CoachingState


def goal_prepare_node(state: CoachingState) -> Dict[str, Any]:
    """사용자에게 단기 코칭 목표를 입력받을 질문을 준비한다.

    다음 노드: wait_for_user.
    """
    prompt = (
        "이번 코칭을 통해 가장 먼저 달라졌으면 하는 점은 무엇인가요? "
        "생각해 둔 기간이 있다면 함께 알려주세요."
    )
    pending = common.interaction("GOAL_INPUT", prompt)
    return {
        "phase": "GOAL",
        "episode_status": "WAITING_USER",
        "pending_interaction": pending,
        "resume_target": "apply_goal",
        "response": prompt,
    }


async def apply_goal_node(state: CoachingState) -> Dict[str, Any]:
    """최초 목표를 생성하거나 사용자 의견으로 기존 목표를 수정한다.

    다음 노드: goal_confirm_prepare.
    """
    answer = common.resume_message(state)
    current_goal = state.get("goal")

    if current_goal:
        fallback = {
            "goal": current_goal,
            "success_criteria": state.get("success_criteria")
            or "보호자가 선택한 행동을 실행하고 변화를 기록하기",
            "time_horizon_days": state.get("time_horizon_days") or 1,
        }
        prompt = GOAL_REVISION_PROMPT.format(
            question=state.get("question", ""),
            current_goal=current_goal,
            current_criteria=fallback["success_criteria"],
            current_days=fallback["time_horizon_days"],
            feedback=answer,
        )
    else:
        fallback = {
            "goal": answer or state.get("question", "현재 돌봄 상황 개선하기"),
            "success_criteria": "보호자가 선택한 행동을 실행하고 변화를 기록하기",
            "time_horizon_days": 1,
        }
        prompt = GOAL_PROMPT.format(
            question=state.get("question", ""),
            answer=answer,
        )

    result = await common.structured_output(prompt, fallback)
    days = max(1, int(result.get("time_horizon_days", 1)))
    return {
        "goal": result.get("goal", fallback["goal"]),
        "success_criteria": result.get("success_criteria", fallback["success_criteria"]),
        "time_horizon_days": days,
        "goal_confirmed": False,
    }


def goal_confirm_prepare_node(state: CoachingState) -> Dict[str, Any]:
    """구조화한 목표를 확인하거나 구체적인 수정 의견을 받는다.

    다음 노드: wait_for_user.
    """
    prompt = (
        f"목표를 ‘{state.get('goal')}’로 정하고, "
        f"성공 기준을 ‘{state.get('success_criteria')}’로 잡아도 될까요? "
        "맞으면 ‘이 목표로 진행’을 선택하고, "
        "수정이 필요하면 원하는 변경 내용을 구체적으로 작성해 주세요."
    )
    pending = common.interaction(
        "GOAL_CONFIRMATION",
        prompt,
        [{"id": "confirm", "label": "이 목표로 진행"}],
        allow_free_text=True,
    )
    return {
        "pending_interaction": pending,
        "episode_status": "WAITING_USER",
        "resume_target": "apply_goal_confirmation",
        "response": prompt,
    }


def apply_goal_confirmation_node(state: CoachingState) -> Command:
    """목표 확인 또는 구체적인 수정 의견을 적용한다.

    분기: 확인 → reality_prepare, 수정 의견 → apply_goal.
    """
    confirmed = common.selected_option(state) == "confirm" or common.resume_message(state) in (
        "확인", "좋아요", "진행",
    )
    return Command(
        update={"goal_confirmed": confirmed},
        goto="reality_prepare" if confirmed else "apply_goal",
    )


GOAL_NODES = {
    "goal_prepare": goal_prepare_node,
    "apply_goal": apply_goal_node,
    "goal_confirm_prepare": goal_confirm_prepare_node,
    "apply_goal_confirmation": apply_goal_confirmation_node,
}
