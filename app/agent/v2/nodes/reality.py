"""GROW의 현재 상황 파악과 확인 노드."""
from __future__ import annotations

from typing import Any, Dict

from langgraph.types import Command

from app.agent.v2.nodes import common
from app.agent.v2.prompts import REALITY_PROMPT
from app.agent.v2.state import CoachingState


def reality_prepare_node(state: CoachingState) -> Dict[str, Any]:
    """현재 상황, 발생 빈도, 기존 시도를 입력받을 질문을 준비한다.

    다음 노드: wait_for_user.
    """
    prompt = "현재 언제, 얼마나 자주 이런 상황이 생기고 지금까지 어떤 방법을 시도했는지 알려주세요."
    pending = common.interaction("REALITY_INPUT", prompt)
    return {
        "phase": "REALITY",
        "episode_status": "WAITING_USER",
        "pending_interaction": pending,
        "resume_target": "apply_reality",
        "response": prompt,
    }


async def apply_reality_node(state: CoachingState) -> Dict[str, Any]:
    """사용자 답변을 현재 상황 요약과 제약조건으로 구조화한다.

    다음 노드: reality_confirm_prepare.
    """
    answer = common.resume_message(state)
    result = await common.structured_output(
        REALITY_PROMPT.format(goal=state.get("goal", ""), answer=answer),
        {"summary": answer, "constraints": []},
    )
    return {
        "reality_summary": result.get("summary", answer),
        "constraints": result.get("constraints", []),
    }


def reality_confirm_prepare_node(state: CoachingState) -> Dict[str, Any]:
    """구조화한 현재 상황이 맞는지 확인할 질문을 준비한다.

    다음 노드: wait_for_user.
    """
    prompt = f"현재 상황을 ‘{state.get('reality_summary')}’로 이해했어요. 맞나요?"
    pending = common.interaction(
        "REALITY_CONFIRMATION",
        prompt,
        [{"id": "confirm", "label": "맞아요"}, {"id": "revise", "label": "추가 설명"}],
    )
    return {
        "pending_interaction": pending,
        "episode_status": "WAITING_USER",
        "resume_target": "apply_reality_confirmation",
        "response": prompt,
    }


def apply_reality_confirmation_node(state: CoachingState) -> Command:
    """현재 상황 확인 또는 추가 설명 요청을 적용한다.

    분기: 확인 → options_prepare, 수정 → reality_prepare.
    """
    confirmed = common.selected_option(state) == "confirm" or common.resume_message(state) in (
        "맞아요", "네", "예",
    )
    return Command(goto="options_prepare" if confirmed else "reality_prepare")


REALITY_NODES = {
    "reality_prepare": reality_prepare_node,
    "apply_reality": apply_reality_node,
    "reality_confirm_prepare": reality_confirm_prepare_node,
    "apply_reality_confirmation": apply_reality_confirmation_node,
}
