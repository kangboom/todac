"""
상태 정의 (AgentState: 질문, 문서, 응급여부)
"""
from typing import TypedDict, List, Optional, Dict, Any, Annotated
import uuid
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from app.dto.qna import QnADoc
from app.dto.rag import RagDoc


class AgentState(TypedDict):
    """LangGraph Agent 상태"""
    # 1. 입력 및 기본 정보
    question: str  # 사용자 질문 (검색용, 재작성될 수 있음)
    previous_question: Optional[str]  # 최초 질문 (재작성 전 보존, 답변 생성 시 참조)
    session_id: uuid.UUID  # 세션 ID
    user_id: uuid.UUID  # 사용자 ID
    baby_info: Optional[Dict[str, Any]]  # 아기 프로필 정보 (교정 연령 포함)
    user_current_info: Optional[str] # 사용자 현재 상황 (HITL 단계에서 수집)
    
    # 2. 대화 이력 (reducer 사용)
    messages: Annotated[List[BaseMessage], add_messages]  # 대화 이력 (BaseMessage 객체)
    
    # 3. 내부 상태 (Internal State) - 노드 간 데이터 전달 및 제어용
    _intent: Optional[str] # 의도 분류 결과 ('relevant' | 'irrelevant')
    
    # 검색된 문서 (평가 전/후)
    _retrieved_docs: Optional[List[RagDoc]] # RAG 검색된 문서 리스트 (DTO 사용)
    _qna_docs: Optional[List[QnADoc]] # QnA 검색된 문서 리스트 (DTO 사용)
    
    # Self-RAG 평가 관련
    _doc_relevance_score: Optional[float]  # 문서 관련성 점수 (0.0 ~ 1.0)
    _doc_relevance_passed: bool  # 문서 관련성 통과 여부 (라우팅용)

    # 4. 출력 및 응답
    response: str # 최종 답변 텍스트
    
    is_retry: bool # 재질문 모드 여부
    is_emergency: bool # 응급 상황 여부
    
    response_time: Optional[float] # 답변 생성 시간 (초)
    
    # 5. 코칭 상태 (Coaching Agent)
    goal: Optional[str]  # 현재 진행 중인 코칭 목표 (예: "수유 자세 교정")
    goal_options: Optional[List[str]]  # LLM이 생성한 목표 후보 리스트 (2~3개)
    _goal_valid: Optional[bool]  # 목표 선택 유효 여부 (라우팅용)
