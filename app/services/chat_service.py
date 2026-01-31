"""
채팅 서비스 (LangGraph 에이전트 실행)
"""
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.models.chat import ChatSession, ChatMessage, MessageRole
from app.models.baby import BabyProfile
from app.agent.graph import get_agent_graph
from app.agent.state import AgentState
from app.core.config import settings
from app.dto.chat import ChatMessageSendResponse, ConversationMessage
from app.dto.baby import AgeInfo, BabyAgentInfo
from typing import Dict, Any, List, Optional, AsyncGenerator
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


def _get_or_create_session(
    db: Session,
    user_id: uuid.UUID,
    baby_id: uuid.UUID,
    session_id: uuid.UUID = None
) -> ChatSession:
    """세션 가져오기 또는 생성"""
    if session_id:
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id
        ).first()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="세션을 찾을 수 없습니다."
            )
        
        return session
    else:
        # 새 세션 생성
        new_session = ChatSession(
            user_id=user_id,
            baby_id=baby_id
        )
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        return new_session


def _get_conversation_history(
    db: Session,
    session_id: uuid.UUID,
    limit: int = 10
) -> List[ConversationMessage]:
    """대화 이력 가져오기"""
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.created_at.desc()).limit(limit).all()
    
    # 최신순으로 정렬 (오래된 것부터)
    messages.reverse()
    
    return [
        ConversationMessage(
            role="user" if msg.role == MessageRole.USER.value else "assistant",
            content=msg.content
        )
        for msg in messages
    ]


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
        session = _get_or_create_session(
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
        conversation_history = _get_conversation_history(
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
        accumulated_response = ""
        
        # [수정] 문서 정보 유실 방지를 위한 별도 캡처 변수
        captured_retrieved_docs = []
        captured_qna_docs = []
        
        # [수정] astream_events를 사용하여 토큰 단위 스트리밍 구현
        async for event in agent_graph.astream_events(initial_state, version="v1"):
            event_type = event.get("event")
            name = event.get("name")
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
                    accumulated_response += chunk_content

            # 2) 상태 추적 (on_chain_end)
            if event_type == "on_chain_end":
                output = data.get("output")
                if output and isinstance(output, dict):
                    # 문서 정보가 있다면 캡처 (덮어쓰기) - 가장 최신의 문서 정보 유지
                    # [수정] 키가 존재하면 값이 비어있어도([]) 갱신하여 초기화를 반영
                    if "_retrieved_docs" in output:
                        captured_retrieved_docs = output["_retrieved_docs"]
                    if "_qna_docs" in output:
                        captured_qna_docs = output["_qna_docs"]
                
                # LangGraph 전체 종료
                if name == "LangGraph":
                    output = data.get("output")
                    if output and isinstance(output, dict):
                        final_state = output
                # 개별 노드 종료 (필요 시)
                elif name == "generate" or name == "intent_classifier":
                    output = data.get("output")
                    if output and isinstance(output, dict):
                        # 부분 상태 업데이트
                        # final_state를 덮어쓰기보다 병합이 안전할 수 있으나, 
                        # LangGraph 노드는 전체 상태를 반환하므로 덮어써도 무방
                        # 단, 가장 마지막에 실행된 노드의 상태가 최종 상태여야 함.
                        # final_state 변수를 계속 갱신하면 됨.
                        final_state = output

        if final_state is initial_state:
             logger.warning("최종 상태를 캡처하지 못했습니다.")
             if accumulated_response:
                 final_state["response"] = accumulated_response
        
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
        # [수정] 캡처된 문서 변수 사용 (final_state에 없어도 복구 가능)
        retrieved_docs = captured_retrieved_docs if captured_retrieved_docs else final_state.get("_retrieved_docs", [])
        
        # [추가] Missing Info 상태라면 문서를 강제로 비움 (안전장치)
        # nodes.py에서 이미 비웠지만, 캡처된 변수에 남아있을 수 있으므로 확인
        # generate 노드가 끝났다면 캡처 변수도 비워져 있어야 정상이지만, 혹시 모를 상황 대비
        # _missing_info가 있거나 의도가 provide_missing_info라면 문서 무시
        if final_state.get("_missing_info") or final_state.get("_intent") == "provide_missing_info":
             retrieved_docs = []
        
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
        # [수정] 캡처된 문서 변수 사용
        qna_docs = captured_qna_docs if captured_qna_docs else final_state.get("_qna_docs", [])
        
        # [추가] Missing Info 상태라면 문서를 강제로 비움
        if final_state.get("_missing_info") or final_state.get("_intent") == "provide_missing_info":
             qna_docs = []
        
        if qna_docs:
            for doc in qna_docs:
                extracted_qna_sources.append({
                    "source_type": "qna",
                    "qna_id": str(_extract_doc_attr(doc, "id", "") or ""),
                    "filename": _extract_doc_attr(doc, "source", "") or "",
                    "category": _extract_doc_attr(doc, "category", "") or "",
                    "question": _extract_doc_attr(doc, "question", "") or "",
                })
        
        # [로깅] 최종 전송할 소스 데이터 확인
        logger.info(f"📤 전송할 RAG 소스: {len(extracted_rag_sources)}개 - {[s.get('filename') for s in extracted_rag_sources]}")
        logger.info(f"📤 전송할 QnA 소스: {len(extracted_qna_sources)}개 - {[s.get('filename') for s in extracted_qna_sources]}")

        combined_sources = []
        combined_sources.extend(extracted_rag_sources)
        combined_sources.extend(extracted_qna_sources)
        
        final_response_text = final_state.get("response", "")
        # 만약 스트리밍된 내용이 있는데 state에 반영 안되었다면 동기화
        if not final_response_text and accumulated_response:
            final_response_text = accumulated_response
            
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


def get_sessions(db: Session, user_id: uuid.UUID, baby_id: uuid.UUID = None) -> List[ChatSession]:
    """사용자의 세션 조회 (baby_id로 필터링 가능)"""
    query = db.query(ChatSession).filter(
        ChatSession.user_id == user_id
    )
    
    if baby_id:
        query = query.filter(ChatSession.baby_id == baby_id)
    
    sessions = query.order_by(ChatSession.updated_at.desc()).all()
    
    return sessions


def get_session_messages(
    db: Session,
    session_id: uuid.UUID,
    user_id: uuid.UUID
) -> List[ChatMessage]:
    """세션의 모든 메시지 조회"""
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user_id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="세션을 찾을 수 없습니다."
        )
    
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.created_at.asc()).all()
    
    return messages


def delete_session(
    db: Session,
    session_id: uuid.UUID,
    user_id: uuid.UUID
) -> None:
    """세션 삭제 (소속된 메시지도 함께 삭제됨)"""
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user_id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="세션을 찾을 수 없습니다."
        )
        
    db.delete(session)
    db.commit()
