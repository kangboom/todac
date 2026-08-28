"""안전 검사, 모드 선택, 사용자 대기와 코칭 동의 노드."""
from __future__ import annotations

import asyncio
from typing import Any, Dict

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

from app.agent.coaching.nodes import common
from app.agent.coaching.prompts import MODE_PROMPT
from app.agent.coaching.state import CoachingState


async def safety_gate_node(state: CoachingState) -> Command:
    """사용자 메시지의 응급 여부를 검사한다.

    분기: 응급 → emergency_response, 일반 → resume_target(기본 mode_router).
    """
    message = common.resume_message(state) or state.get("question", "")
    if await common.is_emergency(message):
        return Command(
            update={"is_emergency": True, "episode_status": "ESCALATED", "pending_interaction": None},
            goto="emergency_response",
        )
    return Command(update={"is_emergency": False}, goto=state.get("resume_target", "mode_router"))


async def mode_router_node(state: CoachingState) -> Command:
    """최초 질문을 코칭 요청과 일반 정보 질문으로 분류한다.

    분기: coaching → consent_prepare, information → general_answer.
    """
    question = state.get("question", "")
    fallback_mode = "coaching" if any(
        token in question for token in ("어떻게", "도와", "고민", "힘들", "방법", "습관")
    ) else "information"
    result = await common.structured_output(MODE_PROMPT.format(question=question), {"mode": fallback_mode})
    mode = result.get("mode", fallback_mode)
    return Command(update={"mode": mode}, goto="consent_prepare" if mode == "coaching" else "general_answer")


async def wait_for_user_node(state: CoachingState, config: RunnableConfig) -> Command:
    """질문을 interrupt로 노출하고, 재개 시 사용자 답변을 상태에 복원한다.

    분기: 사용자 답변으로 재개 → safety_gate.
    """
    payload = state.get("pending_interaction")
    resumed = interrupt(payload)
    resumed_dict = resumed if isinstance(resumed, dict) else {"message": str(resumed)}
    business_context = config.get("configurable", {}).get("business_context") or {}
    return Command(
        update={
            **business_context,
            "latest_resume": resumed_dict,
            "request_id": str(resumed.get("request_id"))
            if isinstance(resumed, dict) and resumed.get("request_id")
            else None,
            "episode_status": "ACTIVE",
        },
        goto="safety_gate",
    )


def consent_prepare_node(state: CoachingState) -> Dict[str, Any]:
    """코칭 진행 동의 질문과 선택지를 준비한다.

    다음 노드: wait_for_user.
    """
    prompt = "정보만 바로 확인할 수도 있고, 목표를 정해 함께 실천할 수도 있어요. 코칭으로 진행할까요?"
    pending = common.interaction(
        "COACHING_CONSENT",
        prompt,
        [{"id": "accept", "label": "코칭으로 진행"}, {"id": "decline", "label": "정보만 보기"}],
        True,
    )
    return {
        "phase": "CONSENT",
        "episode_status": "WAITING_USER",
        "pending_interaction": pending,
        "resume_target": "apply_consent",
        "response": prompt,
    }


def apply_consent_node(state: CoachingState) -> Command:
    """사용자의 코칭 동의 여부를 적용한다.

    분기: 동의 → goal_prepare, 거절 → general_answer.
    """
    accepted = common.selected_option(state) == "accept" or common.resume_message(state) in (
        "코칭으로 진행", "코칭", "네", "예",
    )
    return Command(goto="goal_prepare" if accepted else "general_answer")


async def general_answer_node(state: CoachingState) -> Dict[str, Any]:
    """근거 문서를 검색해 일반 정보 답변을 생성한다.

    다음 노드: END.
    """
    question = state.get("question", "")
    context, source_ids = await asyncio.to_thread(common.retrieve_context, question)
    prompt = f"""미숙아 보호자에게 인증 문서 근거로 간결하게 답하세요. 진단하지 말고 위험하면 의료기관 상담을 안내하세요.
질문: {question}
근거: {context}"""
    answer = await common.generated_text(
        prompt,
        "관련 정보를 확인하기 어렵습니다. 증상이 지속되면 의료진에게 상담해 주세요.",
        stream=True,
    )
    return {
        "response": answer,
        "source_ids": source_ids,
        "episode_status": "CANCELLED",
        "phase": "COMPLETE",
        "pending_interaction": None,
    }


ROUTING_NODES = {
    "safety_gate": safety_gate_node,
    "mode_router": mode_router_node,
    "wait_for_user": wait_for_user_node,
    "consent_prepare": consent_prepare_node,
    "apply_consent": apply_consent_node,
    "general_answer": general_answer_node,
}
