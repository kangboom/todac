"""GROW 코칭 V2 그래프의 실행 상태."""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, NotRequired, Optional, TypedDict

from langgraph.channels import UntrackedValue

from app.agent.state import BaseAgentState


ReviewRoute = Literal[
    "COMPLETED",
    "ADJUST_WILL",
    "CHANGE_OPTION",
    "UPDATE_REALITY",
    "CHANGE_GOAL",
    "EMERGENCY",
]


class InteractionOption(TypedDict):
    id: str
    label: str
    reason: NotRequired[str]


class PendingInteraction(TypedDict):
    id: str
    kind: str
    prompt: str
    options: List[InteractionOption]
    allow_free_text: bool


class CoachingState(BaseAgentState, total=False):
    # 아래 업무 데이터는 요청 실행 중에만 사용하고 체크포인트에는 기록하지 않는다.
    question: Annotated[str, UntrackedValue]
    previous_question: Annotated[str, UntrackedValue]
    session_id: Annotated[Any, UntrackedValue]
    user_id: Annotated[Any, UntrackedValue]
    response: Annotated[str, UntrackedValue]
    is_emergency: Annotated[bool, UntrackedValue]

    episode_id: str
    phase: str
    episode_status: str
    attempt_count: int
    request_id: Annotated[Optional[str], UntrackedValue]
    resume_target: str
    latest_resume: Annotated[Dict[str, Any], UntrackedValue]
    pending_interaction: Annotated[Optional[PendingInteraction], UntrackedValue]

    mode: Annotated[str, UntrackedValue]
    goal: Annotated[Optional[str], UntrackedValue]
    success_criteria: Annotated[Optional[str], UntrackedValue]
    time_horizon_days: Annotated[int, UntrackedValue]
    goal_confirmed: Annotated[bool, UntrackedValue]

    reality_summary: Annotated[Optional[str], UntrackedValue]
    constraints: Annotated[List[str], UntrackedValue]
    action_options: Annotated[List[InteractionOption], UntrackedValue]
    selected_action: Annotated[Optional[str], UntrackedValue]
    action_plan: Annotated[Optional[Dict[str, Any]], UntrackedValue]

    execution_result: Annotated[Optional[str], UntrackedValue]
    barrier: Annotated[Optional[str], UntrackedValue]
    review_route: Annotated[Optional[ReviewRoute], UntrackedValue]
    source_ids: Annotated[List[str], UntrackedValue]
