"""
채팅 서비스 (LangGraph 코칭 에이전트 실행 - HITL 지원)
"""
from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.models.chat import ChatMessage, MessageRole
from app.models.baby import BabyProfile
from app.agent.graph import get_agent_graph
from app.agent.state import AgentState
from app.dto.baby import AgeInfo, BabyAgentInfo
from app.services.chat_repository import get_or_create_session, get_conversation_history
from typing import Any, AsyncGenerator
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import time
import logging
import json
from datetime import date, datetime, timezone

logger = logging.getLogger(__name__)


def _calculate_corrected_age(birth_date: date, due_date: date) -> AgeInfo:
    """
    교정 연령 계산
    교정 연령 = 현재 날짜 - 출산 예정일
    """
    today = date.today()
    corrected_age_days = (today - due_date).days
    corrected_age_months = corrected_age_days / 30.44  # 평균 월 길이
    
    return AgeInfo(
        corrected_age_days=corrected_age_days,
        corrected_age_months=round(corrected_age_months, 1),
        chronological_age_days=(today - birth_date).days,
        chronological_age_months=round((today - birth_date).days / 30.44, 1)
    )


def _prepare_baby_info(baby: BabyProfile) -> BabyAgentInfo:
    """아기 정보를 AgentState에 맞는 형식으로 변환"""
    age_info = _calculate_corrected_age(baby.birth_date, baby.due_date)
    
    return BabyAgentInfo(
        baby_id=str(baby.id),
        name=baby.name,
        birth_date=baby.birth_date.isoformat(),
        due_date=baby.due_date.isoformat(),
        gender=baby.gender,
        birth_weight=baby.birth_weight,
        medical_history=baby.medical_history or [],
        **age_info.model_dump()
    )


def _extract_doc_attr(doc: Any, attr: str, default: Any = "") -> Any:
    """문서 객체 또는 딕셔너리에서 속성 추출"""
    if isinstance(doc, dict):
        return doc.get(attr, default)
    return getattr(doc, attr, default)


async def send_message(
    db: Session,
    user_id: uuid.UUID,
    baby_id: uuid.UUID,
    question: str,
    session_id: uuid.UUID = None
) -> AsyncGenerator[str, None]:
    """
    메시지 전송 및 코칭 에이전트 실행 (토큰 단위 스트리밍 + HITL)
    
    스트리밍 대상: coach_agent, closing 노드의 LLM 응답만 (stream_response 태그)
    
    HITL 흐름:
    1. 첫 메시지: intent → goal_setter → coach_agent → [INTERRUPT]
    2. 후속 메시지: Command(resume=사용자응답) → evaluator → coach_agent/closing
    
    Args:
        db: 데이터베이스 세션
        user_id: 사용자 ID
        baby_id: 아기 ID
        question: 사용자 질문
        session_id: 세션 ID (없으면 새로 생성)
    
    Yields:
        SSE 이벤트 데이터 (JSON 문자열)
    """
    start_time = time.time()
    
    try:
        # 1. 세션 가져오기 또는 생성
        session = get_or_create_session(
            db, user_id, baby_id, session_id
        )
        
        # 2. 아기 정보 가져오기
        baby = db.query(BabyProfile).filter(
            BabyProfile.id == baby_id,
            BabyProfile.user_id == user_id
        ).first()
        
        if not baby:
            yield json.dumps({
                "type": "error",
                "detail": "아기 프로필을 찾을 수 없습니다."
            }, ensure_ascii=False)
            return
        
        # 3. 에이전트 그래프 가져오기 (async)
        agent_graph = await get_agent_graph()
        
        # 4. thread_id 기반 config (체크포인터 상태 관리)
        thread_id = str(session.id)
        config = {"configurable": {"thread_id": thread_id}}
        
        # 5. 체크포인터에서 기존 상태 확인 (코칭 진행 중인지)
        existing_state = await agent_graph.aget_state(config)
        is_resuming = (
            existing_state 
            and existing_state.values 
            and existing_state.values.get("goal_status") == "in_progress"
            and existing_state.next  # interrupt 후 다음 노드가 있는 경우
        )
        
        logger.info(f"========== 😊 에이전트 실행 시작: session_id={session.id}, is_resuming={is_resuming}, question={question[:50]}... ==========")
        
        final_state = {}
        
        if is_resuming:
            # ===== HITL 재개 모드 =====
            # 사용자 응답을 resume 값으로 전달하여 interrupted 그래프를 재개
            logger.info(f"🔄 코칭 루프 재개 (thread_id={thread_id})")
            
            graph_input = Command(
                resume=question,
                update={
                    "question": question,
                    "messages": [HumanMessage(content=question)]
                }
            )
        else:
            # ===== 신규 실행 모드 =====
            # 대화 이력 로드 및 초기 상태 구성
            conversation_history = get_conversation_history(db, session.id)
            
            history_messages = []
            if conversation_history:
                for msg in conversation_history:
                    content = msg.content
                    if msg.role == "user":
                        history_messages.append(HumanMessage(content=content))
                    elif msg.role == "assistant":
                        is_retry = getattr(msg, "is_retry", False)
                        history_messages.append(AIMessage(content=content, additional_kwargs={"is_retry": is_retry}))
            
            history_messages.append(HumanMessage(content=question))
            
            graph_input: AgentState = {
                "question": question,
                "previous_question": question,
                "session_id": session.id,
                "user_id": user_id,
                "messages": history_messages,
                "baby_info": _prepare_baby_info(baby).model_dump(),
                "_retrieved_docs": [],
                "_qna_docs": [],
                "_doc_relevance_score": None,
                "_doc_relevance_passed": False,
                "_missing_info": None,
                "is_retry": False,
                "response": "",
                "is_emergency": False,
                "response_time": None,
                "_intent": None,
                # 코칭 상태 초기값
                "goal": None,
                "coaching_steps": None,
                "current_step_idx": 0,
                "goal_status": None,
                # 목표 승인 상태 초기값
                "_goal_approved": None,
                "_goal_feedback": None
            }
        
        # 6. astream_events로 토큰 단위 스트리밍
        # coach_agent, closing 노드에서 "stream_response" 태그가 붙은 LLM 호출만 스트리밍
        async for event in agent_graph.astream_events(graph_input, config=config, version="v2"):
            event_type = event.get("event")
            data = event.get("data", {})
            tags = event.get("tags", [])
            
            # LLM 토큰 스트리밍 (stream_response 태그 기반 필터링)
            # coach_agent_node, closing_node에서 config={"tags": ["stream_response"]}로 호출한 것만 대상
            if event_type == "on_chat_model_stream" and "stream_response" in tags:
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield json.dumps({
                        "type": "chunk",
                        "content": chunk.content
                    }, ensure_ascii=False)

        # 7. 체크포인터에서 확정된 최종 상태 가져오기
        # (on_chain_end 파싱보다 안정적 — 체크포인터가 보장하는 상태)
        saved_state = await agent_graph.aget_state(config)
        if saved_state and saved_state.values:
            final_state = saved_state.values
        
        # 7. 응답 시간 계산
        response_time = time.time() - start_time
        
        # 8. 사용자 메시지 DB 저장
        user_message = ChatMessage(
            session_id=session.id,
            role=MessageRole.USER.value,
            content=question,
            is_emergency=False,
            created_at=datetime.now(timezone.utc)
        )
        db.add(user_message)
        
        # 9. AI 응답 DB 저장
        extracted_rag_sources = []
        retrieved_docs = final_state.get("_retrieved_docs", [])
        
        if retrieved_docs:
            for doc in retrieved_docs:
                extracted_rag_sources.append({
                    "doc_id": str(_extract_doc_attr(doc, "doc_id", "")),
                    "chunk_index": _extract_doc_attr(doc, "chunk_index", ""),
                    "score": _extract_doc_attr(doc, "score", 0.0),
                    "filename": _extract_doc_attr(doc, "filename", ""),
                    "category": _extract_doc_attr(doc, "category", "")
                })
            
        extracted_qna_sources = []
        qna_docs = final_state.get("_qna_docs", [])
        
        if qna_docs:
            for doc in qna_docs:
                extracted_qna_sources.append({
                    "source_type": "qna",
                    "qna_id": str(_extract_doc_attr(doc, "id", "") or ""),
                    "filename": _extract_doc_attr(doc, "source", "") or "",
                    "category": _extract_doc_attr(doc, "category", "") or "",
                    "question": _extract_doc_attr(doc, "question", "") or "",
                })
        
        combined_sources = []
        combined_sources.extend(extracted_rag_sources)
        combined_sources.extend(extracted_qna_sources)
        
        final_response_text = final_state.get("response", "")

        assistant_message = ChatMessage(
            session_id=session.id,
            role=MessageRole.ASSISTANT.value,
            content=final_response_text,
            is_emergency=final_state.get("is_emergency", False),
            is_retry=final_state.get("is_retry", False),
            rag_sources=combined_sources if combined_sources else None,
            created_at=datetime.now(timezone.utc)
        )
        db.add(assistant_message)
        
        # 10. 세션 정보 업데이트
        session.updated_at = datetime.now()
        db.add(session)

        if not session.title:
            session.title = question[:50]
        
        db.commit()
        
        logger.info(f"에이전트 실행 완료: response_time={response_time:.2f}s")
        
        # 11. 완료 이벤트 전송 (코칭 메타데이터 포함)
        done_event = {
            "type": "done",
            "response": final_response_text,
            "session_id": str(session.id),
            "is_emergency": final_state.get("is_emergency", False),
            "rag_sources": extracted_rag_sources,
            "qna_sources": extracted_qna_sources,
            "response_time": response_time,
            # 코칭 메타데이터
            "coaching": {
                "goal": final_state.get("goal"),
                "goal_status": final_state.get("goal_status"),
                "current_step_idx": final_state.get("current_step_idx", 0),
                "total_steps": len(final_state.get("coaching_steps", []) or []),
                "coaching_steps": final_state.get("coaching_steps"),
                "is_coaching": final_state.get("goal_status") in ("in_progress", "completed", "paused"),
                "awaiting_goal_approval": (
                    final_state.get("goal_status") == "in_progress"
                    and final_state.get("_goal_approved") is None
                ),
            }
        }
        
        yield json.dumps(done_event, ensure_ascii=False)
        
    except HTTPException as he:
        yield json.dumps({
            "type": "error",
            "detail": he.detail
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"에이전트 실행 실패: {str(e)}", exc_info=True)
        db.rollback()
        yield json.dumps({
            "type": "error",
            "detail": f"메시지 처리 중 오류가 발생했습니다: {str(e)}"
        }, ensure_ascii=False)
