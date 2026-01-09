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

        initial_state: AgentState = {
            "question": question,
            "original_question": question,  # 원본 질문 초기화
            "session_id": session.id,
            "user_id": user_id,
            "messages": history_messages,
            "baby_info": _prepare_baby_info(baby).model_dump(),  # DTO -> Dict 변환
            "retrieved_docs": [],
            "rag_retrieval_attempts": 0,
            "min_rag_score": settings.MIN_RAG_SCORE_THRESHOLD,
            "_rag_score_passed": False,  # 초기값: False
            # Self-RAG 관련 필드
            "_doc_relevance_score": None,
            "_doc_relevance_passed": False,
            "_hallucination_score": None,
            "_hallucination_passed": False,
            "_generation_attempts": 0,
            "_max_generation_attempts": 3,  # 최대 생성 시도 횟수
            "response": "",
            "is_emergency": False,
            "rag_sources": None,
            "response_time": None
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
        assistant_message = ChatMessage(
            session_id=session.id,
            role=MessageRole.ASSISTANT.value,
            content=final_state.get("response", ""),
            is_emergency=final_state.get("is_emergency", False),
            rag_sources=final_state.get("rag_sources")
        )
        db.add(assistant_message)
        
        # 9. 세션 제목 업데이트 (첫 메시지인 경우)
        if not session.title:
            session.title = question[:50]  # 첫 50자
            session.updated_at = datetime.now()
        
        db.commit()
        
        logger.info(f"에이전트 실행 완료: response_time={response_time:.2f}s, is_emergency={final_state.get('is_emergency')}")
        
        return ChatMessageSendResponse(
            response=final_state.get("response", ""),
            session_id=str(session.id),
            is_emergency=final_state.get("is_emergency", False),
            rag_sources=final_state.get("rag_sources"),
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
