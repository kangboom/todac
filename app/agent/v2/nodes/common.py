"""코칭 노드가 공유하는 입력 해석, LLM 호출, 근거 검색 유틸리티."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Tuple

from langchain_core.messages import SystemMessage

from app.agent.v2.prompts import SAFETY_PROMPT
from app.agent.v2.state import CoachingState, PendingInteraction
from app.agent.utils import parse_json_from_response
from app.core.llm_factory import get_evaluator_llm, get_generator_llm

logger = logging.getLogger(__name__)

EMERGENCY_KEYWORDS = (
    "숨을 못", "호흡 곤란", "숨이 안", "청색증", "파래", "의식이 없",
    "경련", "축 늘어", "깨워도", "수유를 전혀", "피를 토",
)


def interaction(
    kind: str,
    prompt: str,
    options: List[Dict[str, str]] | None = None,
    allow_free_text: bool = True,
) -> PendingInteraction:
    return {
        "id": str(uuid.uuid4()),
        "kind": kind,
        "prompt": prompt,
        "options": options or [],
        "allow_free_text": allow_free_text,
    }


def resume_message(state: CoachingState) -> str:
    value = state.get("latest_resume") or {}
    return str(value.get("message") or "").strip()


def selected_option(state: CoachingState) -> str:
    value = state.get("latest_resume") or {}
    return str(value.get("selected_option_id") or "").strip()


async def structured_output(prompt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    llm = get_evaluator_llm()
    if not llm:
        return fallback
    try:
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        parsed = parse_json_from_response(response.content)
        return parsed or fallback
    except Exception:
        logger.exception("구조화 LLM 호출 실패")
        return fallback


async def generated_text(prompt: str, fallback: str, stream: bool = False) -> str:
    llm = get_generator_llm()
    if not llm:
        return fallback
    try:
        response = await llm.ainvoke(
            [SystemMessage(content=prompt)],
            config={"tags": ["stream_response"]} if stream else None,
        )
        return response.content.strip() or fallback
    except Exception:
        logger.exception("생성 LLM 호출 실패")
        return fallback


def _tool_artifacts(result: Any) -> List[Dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], list):
        return result[1]
    return []


def retrieve_context(query: str) -> Tuple[str, List[str]]:
    # 일반/응급 분기 테스트에서 검색 인프라가 모듈 import를 막지 않도록 지연 로드한다.
    from app.agent.tools import milvus_knowledge_search, retrieve_qna

    docs: List[Dict[str, Any]] = []
    try:
        docs.extend(_tool_artifacts(retrieve_qna.func(query=query)))
    except Exception:
        logger.exception("QnA 검색 실패")
    try:
        docs.extend(_tool_artifacts(milvus_knowledge_search.func(query=query, top_k=3)))
    except Exception:
        logger.exception("Milvus 검색 실패")

    lines: List[str] = []
    source_ids: List[str] = []
    for doc in docs[:6]:
        content = doc.get("answer") or doc.get("content") or ""
        source = doc.get("source") or doc.get("filename") or "공식 문서"
        identifier = doc.get("doc_id") or doc.get("id") or source
        if content:
            lines.append(f"[{source}] {content[:800]}")
        source_ids.append(str(identifier))
    return "\n\n".join(lines) or "검색된 근거 문서 없음", source_ids


async def is_emergency(message: str) -> bool:
    normalized = message.lower()
    if any(keyword in normalized for keyword in EMERGENCY_KEYWORDS):
        return True
    result = await structured_output(SAFETY_PROMPT.format(message=message), {"emergency": False})
    return bool(result.get("emergency", False))
