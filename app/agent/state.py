"""
상태 정의 (AgentState: 질문, 문서, 응급여부)
"""
from typing import TypedDict, List, Optional, Dict, Any, Annotated
from datetime import date
import uuid
from operator import add
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """LangGraph Agent 상태"""
    # 입력
    question: str  # 사용자 질문 (재작성될 수 있음)
    original_question: Optional[str]  # 원본 질문 (재작성 전 보존)
    session_id: uuid.UUID  # 세션 ID
    user_id: uuid.UUID  # 사용자 ID
    
    # 대화 이력 (reducer 사용)
    messages: Annotated[List[BaseMessage], add_messages]  # 대화 이력 (BaseMessage 객체)
    
    # 아기 정보
    baby_info: Optional[Dict[str, Any]]  # 아기 프로필 정보 (교정 연령 포함)
    
    # RAG 검색 결과
    retrieved_docs: List[Dict[str, Any]]  # 검색된 문서 리스트
    rag_retrieval_attempts: int  # RAG 검색 시도 횟수
    min_rag_score: float  # 최소 RAG 스코어 임계값
    _rag_score_passed: bool  # RAG 스코어 통과 여부 (라우팅용)
    
    # Self-RAG 관련 필드
    _doc_relevance_score: Optional[float]  # 문서 관련성 점수 (0.0 ~ 1.0)
    _doc_relevance_passed: bool  # 문서 관련성 통과 여부 (라우팅용)
    _hallucination_score: Optional[float]  # 환각 점수 (0.0 ~ 1.0, 높을수록 정확)
    _hallucination_passed: bool  # 환각 체크 통과 여부 (라우팅용)
    _generation_attempts: int  # 답변 생성 시도 횟수
    _max_generation_attempts: int  # 최대 생성 시도 횟수
    
    # 응답
    response: str  # 최종 응답 텍스트
    is_emergency: bool  # 응급 상황 여부
    
    # 메타데이터
    rag_sources: Optional[List[Dict[str, Any]]]  # 참조 문서 정보 (doc_id, score 등)
    response_time: Optional[float]  # 응답 시간 (초)
