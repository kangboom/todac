"""
채팅 서비스 (LangGraph 에이전트 실행)
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
import uuid
import time
import logging
import json
from datetime import date, datetime
from langchain_core.messages import HumanMessage, AIMessage

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
    메시지 전송 및 에이전트 실행 (스트리밍)
    
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
        
        # 3. 대화 이력 가져오기
        conversation_history = get_conversation_history(
            db, session.id
        )
        
        # 4. AgentState 초기화
        history_messages = []
        if conversation_history:
            for msg in conversation_history:
                content = msg.content
                if msg.role == "user":
                    history_messages.append(HumanMessage(content=content))
                elif msg.role == "assistant":
                    history_messages.append(AIMessage(content=content))
        
        history_messages.append(HumanMessage(content=question))

        prev_missing_info = session.missing_info if session.missing_info else None
        
        restored_previous_question = question
        if prev_missing_info and isinstance(prev_missing_info, dict):
            restored_previous_question = prev_missing_info.get("pending_question", question)

        initial_state: AgentState = {
            "question": question,
            "previous_question": restored_previous_question,
            "session_id": session.id,
            "user_id": user_id,
            "messages": history_messages,
            "baby_info": _prepare_baby_info(baby).model_dump(),
            "_retrieved_docs": [],
            "_qna_docs": [],
            "_doc_relevance_score": None,
            "_doc_relevance_passed": False,
            "_missing_info": prev_missing_info,
            "is_retry": False,
            "response": "",
            "is_emergency": False,
            "response_time": None,
            "_intent": None
        }
        
        # 5. 에이전트 그래프 가져오기 및 실행 (스트리밍)
        logger.info(f"에이전트 실행 시작: session_id={session.id}, question={question[:50]}...")
        agent_graph = get_agent_graph()
        
        final_state = initial_state
        
        # [수정] astream_events를 사용하여 토큰 단위 스트리밍 구현
        async for event in agent_graph.astream_events(initial_state, version="v1"):
            event_type = event.get("event")
            data = event.get("data", {})
            tags = event.get("tags", [])
            
            # 1) LLM 토큰 스트리밍 (태그 기반 필터링)
            # nodes.py에서 "stream_response" 태그를 단 호출만 스트리밍
            if event_type == "on_chat_model_stream" and "stream_response" in tags:
                chunk_content = data.get("chunk", {}).content
                if chunk_content:
                    yield json.dumps({
                        "type": "chunk",
                        "content": chunk_content
                    }, ensure_ascii=False)

            # 2) 상태 추적 (on_chain_end)
            if event_type == "on_chain_end":
                output = data.get("output")
                if output and isinstance(output, dict):
                    # {노드명: State} 형태인지 확인
                    if output and all(isinstance(v, dict) for v in output.values()):
                        final_state = next(iter(output.values()))
                    else:
                        final_state = output
        
        # 6. 응답 시간 계산
        response_time = time.time() - start_time
        
        # 7. 사용자 메시지 DB 저장
        user_message = ChatMessage(
            session_id=session.id,
            role=MessageRole.USER.value,
            content=question,
            is_emergency=False
        )
        db.add(user_message)
        
        # 8. AI 응답 DB 저장
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
            rag_sources=combined_sources if combined_sources else None
        )
        db.add(assistant_message)
        
        # 9. 세션 정보 업데이트
        session.missing_info = final_state.get("_missing_info")
        session.updated_at = datetime.now()
        
        db.add(session)

        if not session.title:
            session.title = question[:50]
        
        db.commit()
        
        logger.info(f"에이전트 실행 완료: response_time={response_time:.2f}s")
        
        # 완료 이벤트 전송
        yield json.dumps({
            "type": "done",
            "response": final_response_text,
            "session_id": str(session.id),
            "is_emergency": final_state.get("is_emergency", False),
            "rag_sources": extracted_rag_sources,
            "qna_sources": extracted_qna_sources,
            "response_time": response_time
        }, ensure_ascii=False)
        
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
