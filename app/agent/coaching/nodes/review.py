"""실행 결과 검토, 완료 확인과 종료 응답 노드."""
from __future__ import annotations

from typing import Any, Dict

from langgraph.types import Command

from app.agent.coaching.nodes import common
from app.agent.coaching.prompts import REVIEW_PROMPT
from app.agent.coaching.state import CoachingState
from app.core.config import settings


def checkin_prepare_node(state: CoachingState) -> Dict[str, Any]:
    """사용자에게 실행 결과와 어려움을 보고받을 질문을 준비한다.

    다음 노드: wait_for_user.
    """
    prompt = "계획을 실천한 뒤 결과를 알려주세요. 무엇이 달라졌고, 어려웠던 점은 무엇이었나요?"
    pending = common.interaction("CHECK_IN", prompt)
    return {
        "phase": "CHECK_IN",
        "pending_interaction": pending,
        "episode_status": "WAITING_USER",
        "resume_target": "apply_checkin",
        "response": prompt,
    }


def apply_checkin_node(state: CoachingState) -> Dict[str, Any]:
    """사용자의 실행 결과 답변을 검토 상태에 적용한다.

    다음 노드: review.
    """
    return {"execution_result": common.resume_message(state), "phase": "REVIEW"}


async def review_node(state: CoachingState) -> Command:
    """실행 결과를 평가해 완료 또는 재조정할 GROW 단계를 결정한다.

    분기: 완료 → completion_prepare, 응급 → emergency_response,
    반복 한도 도달 → escalated_response, 그 외 → will/options/reality/goal_prepare.
    """
    result_text = state.get("execution_result", "")
    fallback_route = "COMPLETED" if any(
        token in result_text for token in ("달성", "좋아졌", "성공")
    ) else "CHANGE_OPTION"
    result = await common.structured_output(
        REVIEW_PROMPT.format(
            goal=state.get("goal", ""),
            criteria=state.get("success_criteria", ""),
            action=state.get("selected_action", ""),
            result=result_text,
        ),
        {"route": fallback_route, "barrier": "", "reason": ""},
    )
    route = str(result.get("route", fallback_route))
    update = {"review_route": route, "barrier": result.get("barrier") or None}
    if route == "EMERGENCY":
        return Command(update={**update, "is_emergency": True}, goto="emergency_response")
    if route == "COMPLETED":
        return Command(update=update, goto="completion_prepare")
    if state.get("attempt_count", 0) >= settings.COACHING_MAX_ADJUSTMENTS:
        return Command(update=update, goto="escalated_response")
    destinations = {
        "ADJUST_WILL": "will_prepare",
        "CHANGE_OPTION": "options_prepare",
        "UPDATE_REALITY": "reality_prepare",
        "CHANGE_GOAL": "goal_prepare",
    }
    return Command(update=update, goto=destinations.get(route, "options_prepare"))


def completion_prepare_node(state: CoachingState) -> Dict[str, Any]:
    """성공 기준 달성 여부를 최종 확인할 질문을 준비한다.

    다음 노드: wait_for_user.
    """
    prompt = f"설정한 성공 기준 ‘{state.get('success_criteria')}’을 달성한 것으로 정리해도 될까요?"
    pending = common.interaction(
        "COMPLETION_CONFIRMATION",
        prompt,
        [{"id": "confirm", "label": "목표 달성"}, {"id": "continue", "label": "조금 더 해보기"}],
    )
    return {
        "pending_interaction": pending,
        "episode_status": "WAITING_USER",
        "resume_target": "apply_completion",
        "response": prompt,
    }


def apply_completion_node(state: CoachingState) -> Command:
    """사용자의 완료 확인 또는 추가 실행 요청을 적용한다.

    분기: 완료 → complete_response, 계속 실행 → will_prepare.
    """
    confirmed = common.selected_option(state) == "confirm" or common.resume_message(state) in (
        "달성", "완료", "네",
    )
    return Command(goto="complete_response" if confirmed else "will_prepare")


def complete_response_node(state: CoachingState) -> Dict[str, Any]:
    """완료 상태와 최종 코칭 응답을 만든다.

    다음 노드: END.
    """
    response = (
        f"‘{state.get('goal')}’ 목표를 완료했어요. 효과가 있었던 방법을 유지해 보세요. "
        "같은 채팅방에서 새 목표를 시작하거나 다른 질문을 이어갈 수 있어요."
    )
    return {
        "phase": "COMPLETE",
        "episode_status": "COMPLETED",
        "pending_interaction": None,
        "response": response,
    }


def escalated_response_node(state: CoachingState) -> Dict[str, Any]:
    """반복 조정 한도에 도달한 코칭을 중단하고 전문가 상담을 안내한다.

    다음 노드: END.
    """
    response = "여러 차례 계획을 조정했지만 개선이 확인되지 않았어요. 반복을 중단하고 담당 의료진이나 전문가와 상담해 주세요."
    return {
        "phase": "COMPLETE",
        "episode_status": "ESCALATED",
        "pending_interaction": None,
        "response": response,
    }


def emergency_response_node(state: CoachingState) -> Dict[str, Any]:
    """코칭을 즉시 중단하고 응급 대응 안내를 반환한다.

    다음 노드: END.
    """
    response = (
        "지금은 코칭보다 아기의 안전 확인이 우선입니다. 호흡 곤란, 청색증, 의식 저하가 있으면 "
        "즉시 119 또는 가까운 응급실에 연락하고 의료진의 안내를 따라주세요."
    )
    return {
        "phase": "COMPLETE",
        "episode_status": "ESCALATED",
        "pending_interaction": None,
        "is_emergency": True,
        "response": response,
    }


REVIEW_NODES = {
    "checkin_prepare": checkin_prepare_node,
    "apply_checkin": apply_checkin_node,
    "review": review_node,
    "completion_prepare": completion_prepare_node,
    "apply_completion": apply_completion_node,
    "complete_response": complete_response_node,
    "escalated_response": escalated_response_node,
    "emergency_response": emergency_response_node,
}
