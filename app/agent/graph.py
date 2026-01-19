"""
Workflow 정의 (StateGraph, Edge 연결)
"""
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage
from app.agent.state import AgentState
from app.agent.nodes import (
    agent_node,
    grade_documents_node,
    rewrite_query_node,
    generate_node,
    grade_hallucination_node,
    retrieve_qna_node  # [추가]
)
from app.agent.tools import milvus_knowledge_search, report_emergency
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


def should_continue(state: AgentState) -> str:
    """
    Agent Node에서 Tool 호출 여부 결정
    - Tool 호출이 있으면 "tools" (tool 실행)
    - Tool 호출이 없으면:
      - Yellow Mode (QnA >= 0.7): "generate" (QnA 기반 답변 생성)
      - Red Mode (QnA < 0.7): "end" (LLM 직접 답변 완료)
    - 응급 응답이 있으면 "end" (응급 응답 완료)
    """
    # 하지만 메시지 확인을 위해 먼저 변수 할당
    messages = state.get("messages", [])
    if not messages:
        return "end"
    
    last_message = messages[-1]
    
    # 1. Tool 호출 확인
    has_tool_call = False
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        has_tool_call = True
    elif isinstance(last_message, dict) and last_message.get("tool_calls"):
        has_tool_call = True
        
    if has_tool_call:
        logger.info("Tool 호출이 감지되었습니다. Tool 실행으로 진행합니다.")
        return "tools"

    # 2. Generate 진입 조건 확인 (Yellow Mode 또는 응급 상황)
    # 응급 상황이거나 QnA 점수가 높으면 답변 생성 노드로 이동
    qna_score = state.get("qna_score", 0.0)
    is_emergency = state.get("is_emergency", False)
    
    if qna_score >= 0.7 or is_emergency:
        reason = "응급 상황" if is_emergency else f"Yellow Mode (Score: {qna_score:.2f})"
        logger.info(f"📝 {reason}: 답변 생성을 위해 Generate로 이동")
        return "generate"
        
    logger.info("Tool 호출이 없고 Red Mode입니다. 직접 답변 완료.")
    return "end"


def route_after_tools(state: AgentState) -> str:
    """
    Tool 실행 후 라우팅
    - milvus_knowledge_search 실행 결과: "grade_docs"
    - emergency_protocol_handler 실행 결과: "agent" (다시 에이전트로 돌아가서 응답 처리)
    """
    messages = state.get("messages", [])
    if not messages:
        return "agent"
        
    last_message = messages[-1]
    
    # 마지막 메시지가 ToolMessage인 경우
    if isinstance(last_message, ToolMessage):
        tool_name = getattr(last_message, "name", "")
        
        # Tool 이름이 없으면(LangGraph 버전에 따라 다를 수 있음) 내용으로 추론
        if not tool_name:
            content = last_message.content
            if isinstance(content, list): # 검색 결과는 보통 리스트
                tool_name = "milvus_knowledge_search"
        
        # [수정] 검색 툴이 포함되어 있으면 우선적으로 문서 평가로 이동
        if tool_name == "milvus_knowledge_search":
            logger.info("RAG 검색 결과입니다. 문서 평가로 진행합니다.")
            return "grade_docs"
            
    # fallback: 알 수 없는 경우 agent로
    return "agent"


def route_doc_relevance(state: AgentState) -> str:
    """
    문서 관련성 평가 결과에 따른 라우팅
    - 관련성 높음: "generate" (답변 생성)
    - 관련성 낮음: "rewrite" (질문 재구성)
      - 단, 최대 검색 시도 횟수(1회)를 초과하거나 응급 상황인 경우 강제로 "generate"로 이동
    """
    relevance_passed = state.get("_doc_relevance_passed", False)
    rag_retrieval_attempts = state.get("rag_retrieval_attempts", 0)
    is_emergency = state.get("is_emergency", False)
    
    if relevance_passed:
        logger.info("문서 관련성이 높습니다. 답변 생성으로 진행합니다.")
        return "generate"
    
    # [수정] 응급 상황이면 재검색 없이 바로 생성으로 이동
    if is_emergency:
        logger.info("🚨 응급 상황이므로 문서 관련성이 낮아도 바로 답변 생성으로 진행합니다.")
        return "generate"
    
    # [추가] 최대 시도 횟수 초과 체크
    if rag_retrieval_attempts >= 1:  # 최대 1회만 재구성
        logger.warning(f"문서 관련성이 낮지만 최대 검색 시도(1)에 도달하여 답변 생성을 강제합니다.")
        return "generate"
        
    logger.info(f"문서 관련성이 낮습니다 (시도 {rag_retrieval_attempts}). 질문 재구성으로 진행합니다.")
    return "rewrite"


def route_hallucination(state: AgentState) -> str:
    """
    환각 평가 결과에 따른 라우팅
    - 점수 통과: "end" (최종 답변 반환)
    - 점수 미달: "generate" (재생성) 또는 "end" (최대 시도 도달)
    """
    hallucination_passed = state.get("_hallucination_passed", False)
    attempts = state.get("_generation_attempts", 0)
    max_attempts = state.get("_max_generation_attempts", 3)
    
    if hallucination_passed:
        logger.info("환각 평가 통과. 최종 답변 반환.")
        return "end"
    elif attempts < max_attempts:
        logger.warning(f"환각 평가 미통과. 답변 재생성 시도 ({attempts}/{max_attempts})")
        return "generate"
    else:
        logger.warning(f"최대 생성 시도 횟수({max_attempts}) 도달. 현재 답변 반환.")
        return "end"


def route_qna_check(state: AgentState) -> str:
    """
    [Strategy B] QnA 검색 결과에 따른 라우팅 (Green Signal Check)
    - Score >= 0.9: Green -> 바로 생성
    - Score < 0.9: Yellow/Red -> Agent로 이동하여 추가 탐색
    """
    qna_score = state.get("qna_score", 0.0)
    
    if qna_score >= 0.9:
        logger.info(f"🚀 Green Mode (Score: {qna_score:.2f}): QnA 결과로 바로 답변 생성")
        return "generate"
    else:
        logger.info(f"🚦 Score {qna_score:.2f}: Agent로 이동하여 추가 탐색 (Yellow/Red)")
        return "agent"


def create_agent_graph():
    """
    LangGraph 에이전트 그래프 생성 (Self-RAG 구조)
    
    플로우:
    START → retrieve_qna → [Green?]
      - Yes → generate
      - No (Yellow/Red) → agent → [tool 호출?]
        - Yes → tools → grade_docs → [관련성 높음?]
          - Yes → generate → grade_hallucination → [점수 통과?]
            - Yes → END
            - No → generate (재시도) 또는 END (최대 시도)
          - No → rewrite → agent
        - No → [Yellow?]
          - Yes (Yellow) -> generate (QnA 기반 생성)
          - No (Red) -> END (직접 답변)
    """
    # Tool 정의 (모든 tool을 LLM에 제공)
    tools = [
        milvus_knowledge_search,  # RAG 검색 tool
        report_emergency,         # 응급 상태 보고 tool
    ]
    
    # StateGraph 생성
    workflow = StateGraph(AgentState)
    
    # ToolNode 생성 (Tool 실행 노드)
    tool_node = ToolNode(tools)
    
    # 노드 추가
    workflow.add_node("retrieve_qna", retrieve_qna_node) # [추가] QnA 검색
    workflow.add_node("agent", agent_node)  # 질문 분석/도구 호출 결정
    workflow.add_node("tools", tool_node)  # ToolNode: Vector DB 검색
    workflow.add_node("grade_docs", grade_documents_node)  # 검색 결과 관련성 평가
    workflow.add_node("rewrite", rewrite_query_node)  # 질문 재구성
    workflow.add_node("generate", generate_node)  # 답변 생성
    workflow.add_node("grade_hallucination", grade_hallucination_node)  # 환각 및 정확도 체크
    
    # 엣지 연결
    
    # 1. START -> QnA 검색 (항상 먼저 실행)
    workflow.add_edge(START, "retrieve_qna")
    
    # 2. QnA 결과 분기 (Green vs Yellow/Red)
    workflow.add_conditional_edges(
        "retrieve_qna",
        route_qna_check,
        {
            "generate": "generate",
            "agent": "agent"
        }
    )
    
    # 3. Agent -> Tools 결정
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",  # ToolNode: Tool 실행 및 ToolMessage 자동 추가
            "end": END,  # 도구 없이 직접 답변 완료 (Red Mode)
            "generate": "generate" # [추가] Yellow Mode (QnA 반영)
        }
    )
    
    # 4. Tools 실행 후 -> 라우팅 (milvus_knowledge_search는 grade_docs, 나머지는 agent)
    workflow.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "grade_docs": "grade_docs",
            "agent": "agent"
        }
    )
    
    # 5. grade_docs -> generate (관련성 높음) 또는 rewrite (관련성 낮음)
    workflow.add_conditional_edges(
        "grade_docs",
        route_doc_relevance,
        {
            "generate": "generate",  # 답변 생성
            "rewrite": "rewrite"  # 질문 재구성
        }
    )
    
    # 6. rewrite -> agent (재검색을 위해 다시 agent로)
    workflow.add_edge("rewrite", "agent")
    
    # 7. generate -> grade_hallucination
    workflow.add_edge("generate", "grade_hallucination")
    
    # 8. grade_hallucination -> END (점수 통과) 또는 generate (재생성) 또는 END (최대 시도)
    workflow.add_conditional_edges(
        "grade_hallucination",
        route_hallucination,
        {
            "end": END,  # 최종 답변 반환
            "generate": "generate"  # 답변 재생성
        }
    )
    
    # 그래프 컴파일
    app = workflow.compile()
    
    return app


# 전역 그래프 인스턴스 (한 번만 생성)
_agent_graph = None


def get_agent_graph():
    """
    에이전트 그래프 인스턴스 가져오기 (싱글톤)
    """
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = create_agent_graph()
    return _agent_graph
