"""GROW의 선택지 생성, 실행 계획, 자신감 확인 노드."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

from langgraph.types import Command

from app.agent.v2.nodes import common
from app.agent.v2.prompts import (
    OPTIONS_PROMPT,
    OPTIONS_REGENERATION_PROMPT,
    OPTIONS_VALIDATION_PROMPT,
    PLAN_PROMPT,
)
from app.agent.v2.state import CoachingState
from app.core.config import settings

logger = logging.getLogger(__name__)

MAX_OPTION_REGENERATION_ATTEMPTS = 1


def _normalize_options(raw_options: Any) -> List[Dict[str, str]]:
    """LLM이 생성한 선택지를 사용자에게 전달할 공통 형식으로 정리한다."""
    if not isinstance(raw_options, list):
        return []
    options: List[Dict[str, str]] = []
    for index, item in enumerate(raw_options[:3]):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        options.append({
            "id": str(item.get("id") or f"option-{index + 1}"),
            "label": label,
            "reason": str(item.get("reason") or "").strip(),
        })
    return options


def _validate_option_structure(
    medical_context: str,
    options: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """LLM 호출 없이 확인할 수 있는 선택지 형식 오류를 반환한다."""
    issues: List[Dict[str, Any]] = []
    if not medical_context.strip():
        issues.append({
            "option_id": "",
            "codes": ["MEDICAL_CONTEXT_MISSING"],
            "feedback": "현재 상황을 설명하는 의학적 배경이 없습니다.",
        })
    if not 2 <= len(options) <= 3:
        issues.append({
            "option_id": "",
            "codes": ["INVALID_OPTION_COUNT"],
            "feedback": "행동 선택지는 2개 또는 3개여야 합니다.",
        })

    seen_labels = set()
    for option in options:
        option_id = option["id"]
        if not option["reason"]:
            issues.append({
                "option_id": option_id,
                "codes": ["REASON_MISSING"],
                "feedback": "선택지를 만든 이유가 없습니다.",
            })
        normalized_label = option["label"].casefold()
        if normalized_label in seen_labels:
            issues.append({
                "option_id": option_id,
                "codes": ["DUPLICATE_OPTION"],
                "feedback": "다른 선택지와 내용이 중복됩니다.",
            })
        seen_labels.add(normalized_label)
    return issues


async def _validate_generated_options(
    *,
    question: str,
    goal: str,
    reality: str,
    context: str,
    medical_context: str,
    options: List[Dict[str, str]],
) -> Dict[str, Any]:
    """선택지의 형식과 목표 일치성, 근거성, 안전성을 검수한다."""
    structure_issues = _validate_option_structure(medical_context, options)
    if structure_issues:
        return {"valid": False, "issues": structure_issues}

    candidate = {
        "medical_context": medical_context,
        "options": options,
    }
    fallback = {
        "valid": False,
        "issues": [{
            "option_id": "",
            "codes": ["VALIDATION_UNAVAILABLE"],
            "feedback": "선택지 검수 결과를 확인하지 못했습니다.",
        }],
    }
    result = await common.structured_output(
        OPTIONS_VALIDATION_PROMPT.format(
            question=question,
            goal=goal,
            reality=reality,
            context=context,
            candidate=json.dumps(candidate, ensure_ascii=False),
        ),
        fallback,
    )
    issues = result.get("issues")
    if not isinstance(issues, list):
        issues = fallback["issues"]
    return {
        "valid": result.get("valid") is True and not issues,
        "issues": issues,
    }


async def _regenerate_options(
    *,
    question: str,
    goal: str,
    reality: str,
    context: str,
    previous: Dict[str, Any],
    validation: Dict[str, Any],
    fallback: Dict[str, Any],
) -> Dict[str, Any]:
    """검수 피드백을 반영해 행동 선택지 전체를 다시 생성한다."""
    return await common.structured_output(
        OPTIONS_REGENERATION_PROMPT.format(
            question=question,
            goal=goal,
            reality=reality,
            context=context,
            previous=json.dumps(previous, ensure_ascii=False),
            validation=json.dumps(validation, ensure_ascii=False),
        ),
        fallback,
    )


async def options_prepare_node(state: CoachingState) -> Dict[str, Any]:
    """의학적 배경과 함께 근거 기반 행동 선택지를 준비한다.

    다음 노드: wait_for_user.
    """
    question = state.get("previous_question") or state.get("question", "")
    goal = state.get("goal", "")
    reality = state.get("reality_summary", "")
    query = f"{question} {goal} {reality}"
    context, source_ids = await asyncio.to_thread(common.retrieve_context, query)
    fallback = {
        "medical_context": (
            "아기의 상태 변화는 한 가지 정보만으로 판단하기보다 "
            "돌봄 상황과 아기의 반응을 함께 관찰하는 것이 도움이 됩니다."
        ),
        "options": [
            {
                "id": "option-1",
                "label": "상황과 반응을 짧게 기록해 보기",
                "reason": "현재 나타나는 변화를 같은 기준으로 비교해 다음 행동을 판단하기 위해서입니다.",
            },
            {
                "id": "option-2",
                "label": "한 번에 한 가지 환경 조건만 조정해 보기",
                "reason": "여러 조건을 동시에 바꾸지 않아야 어떤 변화가 아기의 반응과 관련 있는지 관찰하기 쉽기 때문입니다.",
            },
        ],
    }
    result = await common.structured_output(
        OPTIONS_PROMPT.format(
            question=question,
            goal=goal,
            reality=reality,
            context=context,
        ),
        fallback,
    )
    accepted_result: Dict[str, Any] | None = None
    validation: Dict[str, Any] = {"valid": False, "issues": []}
    for attempt in range(MAX_OPTION_REGENERATION_ATTEMPTS + 1):
        medical_context = str(result.get("medical_context") or "").strip()
        options = _normalize_options(result.get("options"))
        logger.info(
            "선택지 평가 진입: episode_id=%s, attempt=%d, option_count=%d",
            state.get("episode_id"),
            attempt + 1,
            len(options),
        )
        validation = await _validate_generated_options(
            question=question,
            goal=goal,
            reality=reality,
            context=context,
            medical_context=medical_context,
            options=options,
        )
        if validation["valid"]:
            accepted_result = {
                "medical_context": medical_context,
                "options": options,
            }
            break
        if attempt < MAX_OPTION_REGENERATION_ATTEMPTS:
            logger.info(
                "선택지 재생성 진입: episode_id=%s, attempt=%d, issue_count=%d",
                state.get("episode_id"),
                attempt + 1,
                len(validation.get("issues") or []),
            )
            result = await _regenerate_options(
                question=question,
                goal=goal,
                reality=reality,
                context=context,
                previous={"medical_context": medical_context, "options": options},
                validation=validation,
                fallback=fallback,
            )

    if accepted_result is None:
        medical_context = fallback["medical_context"]
        options = _normalize_options(fallback["options"])
    else:
        medical_context = accepted_result["medical_context"]
        options = accepted_result["options"]
    option_explanations = "\n".join(
        f"{i}. {option['label']}\n   - 제안 이유: {option['reason']}"
        for i, option in enumerate(options, start=1)
    )
    prompt = (
        f"{medical_context}\n\n"
        f"현재 목표: {goal}\n"
        f"현재 상황: {reality}\n\n"
        f"{option_explanations}\n\n"
        "어떤 방법을 먼저 시도해 볼까요?"
    )
    pending = common.interaction("ACTION_SELECTION", prompt, options)
    return {
        "phase": "OPTIONS",
        "action_options": options,
        "source_ids": source_ids,
        "pending_interaction": pending,
        "episode_status": "WAITING_USER",
        "resume_target": "apply_option",
        "response": prompt,
    }


def apply_option_node(state: CoachingState) -> Dict[str, Any]:
    """사용자가 선택하거나 직접 입력한 행동을 상태에 적용한다.

    다음 노드: will_prepare.
    """
    selected_id = common.selected_option(state)
    selected_action = common.resume_message(state)
    for option in state.get("action_options", []):
        if option["id"] == selected_id:
            selected_action = option["label"]
            break
    return {"selected_action": selected_action}


async def will_prepare_node(state: CoachingState) -> Dict[str, Any]:
    """선택 행동을 시점, 기간, 관찰 항목이 포함된 실행 계획으로 만든다.

    다음 노드: wait_for_user.
    """
    fallback = {
        "when": "다음 돌봄 상황부터",
        "duration": "1일",
        "observation": "아기의 반응과 실행 결과",
    }
    plan = await common.structured_output(
        PLAN_PROMPT.format(
            goal=state.get("goal", ""),
            reality=state.get("reality_summary", ""),
            action=state.get("selected_action", ""),
        ),
        fallback,
    )
    prompt = (
        f"‘{state.get('selected_action')}’을 {plan.get('when')}, {plan.get('duration')} 동안 시도하고 "
        f"{plan.get('observation')}을 관찰하는 계획으로 진행할까요?"
    )
    pending = common.interaction(
        "PLAN_CONFIRMATION",
        prompt,
        [{"id": "confirm", "label": "이 계획으로 진행"}, {"id": "revise", "label": "다른 방법 선택"}],
    )
    return {
        "phase": "WILL",
        "action_plan": plan,
        "pending_interaction": pending,
        "episode_status": "WAITING_USER",
        "resume_target": "apply_plan_confirmation",
        "response": prompt,
    }


def apply_plan_confirmation_node(state: CoachingState) -> Command:
    """실행 계획 확인 또는 다른 행동 선택 요청을 적용한다.

    분기: 확인 → confidence_prepare, 수정 → options_prepare.
    """
    confirmed = common.selected_option(state) == "confirm" or common.resume_message(state) in (
        "진행", "좋아요", "확인",
    )
    return Command(goto="confidence_prepare" if confirmed else "options_prepare")


def confidence_prepare_node(state: CoachingState) -> Dict[str, Any]:
    """실행 계획에 대한 0~10점 자신감 질문을 준비한다.

    다음 노드: wait_for_user.
    """
    prompt = "이 계획을 실제로 해볼 수 있다는 자신감은 0점부터 10점 중 몇 점인가요?"
    options = [{"id": f"score-{score}", "label": str(score)} for score in range(0, 11)]
    pending = common.interaction("CONFIDENCE_RATING", prompt, options, False)
    return {
        "pending_interaction": pending,
        "episode_status": "WAITING_USER",
        "resume_target": "apply_confidence",
        "response": prompt,
    }


def apply_confidence_node(state: CoachingState) -> Command:
    """자신감 점수를 적용하고 실행 단계 진입 여부를 결정한다.

    분기: 기준 미만 → options_prepare, 기준 이상 → checkin_prepare.
    """
    raw = common.selected_option(state).replace("score-", "") or common.resume_message(state)
    try:
        score = max(0, min(10, int(raw)))
    except ValueError:
        score = 0
    if score < settings.COACHING_CONFIDENCE_THRESHOLD:
        return Command(update={"confidence_score": score}, goto="options_prepare")
    return Command(
        update={"confidence_score": score, "attempt_count": state.get("attempt_count", 0) + 1},
        goto="checkin_prepare",
    )


ACTION_NODES = {
    "options_prepare": options_prepare_node,
    "apply_option": apply_option_node,
    "will_prepare": will_prepare_node,
    "apply_plan_confirmation": apply_plan_confirmation_node,
    "confidence_prepare": confidence_prepare_node,
    "apply_confidence": apply_confidence_node,
}
