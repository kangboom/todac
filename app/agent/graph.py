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
    grade_hallucination_node
)
from app.agent.tools import milvus_knowledge_search, emergency_protocol_handler
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


def should_continue(state: AgentState) -> str:
    """
    Agent Node에서 Tool 호출 여부 결정
    - Tool 호출이 있으면 "tools" (tool 실행)
    - Tool 호출이 없으면 "end" (직접 답변 완료)
    - 응급 응답이 있으면 "end" (응급 응답 완료)
    """
    # 응급 응답이 이미 생성되었으면 종료
    if state.get("response") and state.get("is_emergency"):
        logger.info("응급 응답이 생성되었습니다. 종료합니다.")
        return "end"
    
    messages = state.get("messages", [])
    if not messages:
        return "end"
    
    last_message = messages[-1]
    
    # LangChain AIMessage 객체인 경우
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        logger.info("Tool 호출이 감지되었습니다. Tool 실행으로 진행합니다.")
        return "tools"
    # 딕셔너리 형식인 경우
    elif isinstance(last_message, dict) and last_message.get("tool_calls"):
        logger.info("Tool 호출이 감지되었습니다. Tool 실행으로 진행합니다.")
        return "tools"
    else:
        logger.info("Tool 호출이 없습니다. 직접 답변 완료.")
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
            elif isinstance(content, str) and ("응급" in content or "119" in content):
                tool_name = "emergency_protocol_handler"
        
        if tool_name == "milvus_knowledge_search":
            logger.info("RAG 검색 결과입니다. 문서 평가로 진행합니다.")
            return "grade_docs"
        elif tool_name == "emergency_protocol_handler":
            logger.info("응급 프로토콜 결과입니다. 에이전트로 복귀합니다.")
            return "agent"
            
    # fallback: 알 수 없는 경우 agent로
    return "agent"


def route_doc_relevance(state: AgentState) -> str:
    """
    문서 관련성 평가 결과에 따른 라우팅
    - 관련성 높음: "generate" (답변 생성)
    - 관련성 낮음: "rewrite" (질문 재구성)
    """
    relevance_passed = state.get("_doc_relevance_passed", False)
    
    if relevance_passed:
        logger.info("문서 관련성이 높습니다. 답변 생성으로 진행합니다.")
        return "generate"
    else:
        logger.info("문서 관련성이 낮습니다. 질문 재구성으로 진행합니다.")
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


def create_agent_graph():
    """
    LangGraph 에이전트 그래프 생성 (Self-RAG 구조)
    
    플로우:
    START → agent → [tool 호출?]
      - Yes → tools → grade_docs → [관련성 높음?]
        - Yes → generate → grade_hallucination → [점수 통과?]
          - Yes → END
          - No → generate (재시도) 또는 END (최대 시도)
        - No → rewrite → agent
      - No → END (직접 답변)
    """
    # Tool 정의 (모든 tool을 LLM에 제공)
    tools = [
        milvus_knowledge_search,  # RAG 검색 tool
        emergency_protocol_handler  # 응급 처리 tool
    ]
    
    # StateGraph 생성
    workflow = StateGraph(AgentState)
    
    # ToolNode 생성 (Tool 실행 노드)
    tool_node = ToolNode(tools)
    
    # 노드 추가
    workflow.add_node("agent", agent_node)  # 질문 분석/도구 호출 결정
    workflow.add_node("tools", tool_node)  # ToolNode: Vector DB 검색
    workflow.add_node("grade_docs", grade_documents_node)  # 검색 결과 관련성 평가
    workflow.add_node("rewrite", rewrite_query_node)  # 질문 재구성
    workflow.add_node("generate", generate_node)  # 답변 생성
    workflow.add_node("grade_hallucination", grade_hallucination_node)  # 환각 및 정확도 체크
    
    # 엣지 연결
    # 시작점: agent
    workflow.add_edge(START, "agent")
    
    # agent → tools (tool 호출 시) 또는 END (도구 없이 직접 답변)
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",  # ToolNode: Tool 실행 및 ToolMessage 자동 추가
            "end": END  # 도구 없이 직접 답변 완료 또는 응급 응답
        }
    )
    
    # tools 실행 후 → 라우팅 (milvus_knowledge_search는 grade_docs, 나머지는 agent)
    workflow.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "grade_docs": "grade_docs",
            "agent": "agent"
        }
    )
    
    # grade_docs → generate (관련성 높음) 또는 rewrite (관련성 낮음)
    workflow.add_conditional_edges(
        "grade_docs",
        route_doc_relevance,
        {
            "generate": "generate",  # 답변 생성
            "rewrite": "rewrite"  # 질문 재구성
        }
    )
    
    # rewrite → agent (재검색을 위해 다시 agent로)
    workflow.add_edge("rewrite", "agent")
    
    # generate → grade_hallucination
    workflow.add_edge("generate", "grade_hallucination")
    
    # grade_hallucination → END (점수 통과) 또는 generate (재생성) 또는 END (최대 시도)
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
