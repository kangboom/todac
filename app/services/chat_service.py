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
from typing import Dict, Any, List, Optional
import uuid
import time
import logging
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


def send_message(
    db: Session,
    user_id: uuid.UUID,
    baby_id: uuid.UUID,
    question: str,
    session_id: uuid.UUID = None
) -> ChatMessageSendResponse:
    """
    메시지 전송 및 에이전트 실행
    
    Args:
        db: 데이터베이스 세션
        user_id: 사용자 ID
        baby_id: 아기 ID
        question: 사용자 질문
        session_id: 세션 ID (없으면 새로 생성)
    
    Returns:
        응답 정보 DTO
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
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="아기 프로필을 찾을 수 없습니다."
            )
        
        # 3. 대화 이력 가져오기
        conversation_history = _get_conversation_history(
            db, session.id
        )
        
        # 4. AgentState 초기화
        # DB 히스토리를 LangChain Message 객체로 변환
        history_messages = []
        if conversation_history:
            # _get_conversation_history가 최신순(오래된 것 -> 최신 것)으로 이미 반환함
            for msg in conversation_history:
                content = msg.content
                if msg.role == "user":
                    history_messages.append(HumanMessage(content=content))
                elif msg.role == "assistant":
                    history_messages.append(AIMessage(content=content))
        
        # 현재 질문 추가
        history_messages.append(HumanMessage(content=question))

        # 세션에서 이전 턴의 missing_info 불러오기
        prev_missing_info = session.missing_info if session.missing_info else None
        
        # [수정] 이전 턴의 최초 질문이 있다면 복원 (없으면 현재 질문 사용)
        restored_previous_question = question
        if prev_missing_info and isinstance(prev_missing_info, dict):
            restored_previous_question = prev_missing_info.get("pending_question", question)

        initial_state: AgentState = {
            "question": question,
            "previous_question": restored_previous_question,  # [수정] 복원된 질문 적용
            "session_id": session.id,
            "user_id": user_id,
            "messages": history_messages,
            "baby_info": _prepare_baby_info(baby).model_dump(),  # DTO -> Dict 변환
            "_retrieved_docs": [],
            "_qna_docs": [],
            # Self-RAG 관련 필드
            "_doc_relevance_score": None,
            "_doc_relevance_passed": False,
            "_missing_info": prev_missing_info, # [수정] DB에서 복원
            "is_retry": False, # 기본값 False
            "response": "",
            "is_emergency": False,
            "response_time": None,
            "_intent": None
        }
        
        # 5. 에이전트 그래프 가져오기 및 실행
        logger.info(f"에이전트 실행 시작: session_id={session.id}, question={question[:50]}...")
        agent_graph = get_agent_graph()  # 여기서 그래프 인스턴스 생성/가져오기
        final_state = agent_graph.invoke(initial_state)
        
        # 6. 응답 시간 계산
        response_time = time.time() - start_time
        final_state["response_time"] = response_time
        
        # 7. 사용자 메시지 DB 저장
        user_message = ChatMessage(
            session_id=session.id,
            role=MessageRole.USER.value,
            content=question,
            is_emergency=False
        )
        db.add(user_message)
        
        # 8. AI 응답 DB 저장
        # 문서 객체에서 소스 정보 추출
        extracted_rag_sources = []
        retrieved_docs = final_state.get("_retrieved_docs", [])
        for doc in retrieved_docs:
            extracted_rag_sources.append({
                "doc_id": str(getattr(doc, "doc_id", "")),
                "chunk_index": getattr(doc, "chunk_index", ""),
                "score": getattr(doc, "score", 0.0),
                "filename": getattr(doc, "filename", ""),
                "category": getattr(doc, "category", "")
            })
            
        extracted_qna_sources = []
        qna_docs = final_state.get("_qna_docs", [])
        for doc in qna_docs:
            extracted_qna_sources.append({
                "source_type": "qna",
                "qna_id": str(getattr(doc, "id", "") or ""),
                "filename": getattr(doc, "source", "") or "",
                "category": getattr(doc, "category", "") or "",
                "question": getattr(doc, "question", "") or "",
            })
            
        # DB에는 rag_sources 컬럼 하나뿐이므로 합쳐서 저장 (데이터 유실 방지)
        combined_sources = []
        combined_sources.extend(extracted_rag_sources)
        combined_sources.extend(extracted_qna_sources)

        assistant_message = ChatMessage(
            session_id=session.id,
            role=MessageRole.ASSISTANT.value,
            content=final_state.get("response", ""),
            is_emergency=final_state.get("is_emergency", False),
            rag_sources=combined_sources if combined_sources else None
        )
        db.add(assistant_message)
        
        # 9. 세션 정보 업데이트
        # 다음 턴을 위해 missing_info 저장 (없으면 None)
        session.missing_info = final_state.get("_missing_info")
        session.updated_at = datetime.now() # updated_at 갱신

        # 세션 제목 업데이트 (첫 메시지인 경우)
        if not session.title:
            session.title = question[:50]  # 첫 50자
        
        db.commit()
        
        logger.info(f"에이전트 실행 완료: response_time={response_time:.2f}s, is_emergency={final_state.get('is_emergency')}")
        
        return ChatMessageSendResponse(
            response=final_state.get("response", ""),
            session_id=str(session.id),
            is_emergency=final_state.get("is_emergency", False),
            rag_sources=extracted_rag_sources,
            qna_sources=extracted_qna_sources,
            response_time=response_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"에이전트 실행 실패: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"메시지 처리 중 오류가 발생했습니다: {str(e)}"
        )


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
