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
from typing import Any, AsyncGenerator, Dict, List, Tuple
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import time
import asyncio
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


def _load_session_data(db: Session, user_id: uuid.UUID, baby_id: uuid.UUID, session_id: uuid.UUID = None) -> Tuple:
    """동기 DB 작업: 세션, 아기 정보, 대화 이력 로드 (to_thread로 호출)"""
    session = get_or_create_session(db, user_id, baby_id, session_id)
    
    baby = db.query(BabyProfile).filter(
        BabyProfile.id == baby_id,
        BabyProfile.user_id == user_id
    ).first()
    
    return session, baby


def _load_conversation_history(db: Session, session_id: uuid.UUID) -> List:
    """동기 DB 작업: 대화 이력 로드 (to_thread로 호출)"""
    return get_conversation_history(db, session_id)


def _save_results_to_db(
    db: Session,
    session,
    question: str,
    final_state: Dict,
) -> Tuple[str, List[Dict], List[Dict]]:
    """동기 DB 작업: 메시지 저장 및 커밋 (to_thread로 호출)"""
    # 사용자 메시지 저장
    user_message = ChatMessage(
        session_id=session.id,
        role=MessageRole.USER.value,
        content=question,
        is_emergency=False,
        created_at=datetime.now(timezone.utc)
    )
    db.add(user_message)
    
    # RAG 소스 추출
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
    
    # QnA 소스 추출
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
    
    combined_sources = extracted_rag_sources + extracted_qna_sources
    final_response_text = final_state.get("response", "")
    
    # AI 응답 저장
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
    
    # 세션 업데이트
    session.updated_at = datetime.now()
    db.add(session)
    if not session.title:
        session.title = question[:50]
    
    db.commit()
    
    return final_response_text, extracted_rag_sources, extracted_qna_sources


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
    1. 첫 메시지: intent → ask_situation → [INTERRUPT 1: 상황 답변 대기]
    2. 상황 답변: Command(resume) → goal_options → [INTERRUPT 2: 목표 선택 대기]
    3. 목표 선택: Command(resume) → research_agent → evaluate_docs → response_node
    
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
        # 1. 세션 및 아기 정보 로드 (동기 DB → to_thread)
        session, baby = await asyncio.to_thread(
            _load_session_data, db, user_id, baby_id, session_id
        )
        
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
        
        # 5. 체크포인터에서 기존 상태 확인 
        existing_state = await agent_graph.aget_state(config)
        
        # interrupt 상태(next가 존재)라면 무조건 재개
        is_resuming = (
            existing_state 
            and existing_state.next 
            and len(existing_state.next) > 0
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
                    "messages": [HumanMessage(content=question)]
                }
            )
        else:
            # ===== 신규 실행 모드 =====
            # 대화 이력 로드 및 초기 상태 구성
            # 대화 이력 로드 (동기 DB → to_thread)
            conversation_history = await asyncio.to_thread(
                _load_conversation_history, db, session.id
            )
            
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
                "response": "",
                "is_emergency": False,
                "response_time": None,
                "_intent": None,
                "goal": None,
                "goal_options": None
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
        saved_state = await agent_graph.aget_state(config)
        if saved_state and saved_state.values:
            final_state = saved_state.values
        
        # 8. 응답 시간 계산
        response_time = time.time() - start_time
        
        # 9. DB 저장 (동기 DB → to_thread)
        final_response_text, extracted_rag_sources, extracted_qna_sources = await asyncio.to_thread(
            _save_results_to_db, db, session, question, final_state
        )
        
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
            "coaching": {
                "goal": final_state.get("goal"),
                "goal_options": final_state.get("goal_options")
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
