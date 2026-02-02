"""
Workflow 정의 (StateGraph, Edge 연결)
"""
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage
from app.agent.state import AgentState
from app.agent.nodes import (
    agent_node,
    evaluate_node,
    generate_node,
    intent_classifier_node,
    emergency_response_node, # [추가]
)
from app.agent.tools import milvus_knowledge_search, retrieve_qna
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


def route_intent(state: AgentState) -> str:
    """
    의도 분류 결과에 따른 라우팅
    - relevant: "agent" (기존 플로우 시작)
    - irrelevant: END (단순 응답 후 종료)
    - provide_missing_info: "create_query_from_info" (부족한 정보 반영하여 질문 생성)
    """
    intent = state.get("_intent", "relevant")
    
    if intent == "emergency":
        logger.info("🚨 응급 상황 감지 -> Emergency Fast-Track 진입")
        return "emergency_response"

    if intent == "irrelevant":
        logger.info("🚫 질문이 아기 돌봄과 관련이 없습니다 -> 단순 응답 후 종료")
        return END
        
    if intent == "provide_missing_info":
        logger.info("ℹ️ 부족했던 정보 제공 확인 -> 질문 재생성(create_query_from_info)으로 진행")
        return "create_query_from_info"
    
    logger.info("✅ 질문이 관련성이 있습니다 -> agent 노드 진입")
    return "agent"


def should_continue(state: AgentState) -> str:
    """
    Agent Node에서 Tool 호출 여부 결정
    - Tool 호출이 있으면 "tools" (tool 실행)
    - Tool 호출이 없고, 참고할 문서(retrieved_docs/qna_docs)가 있으면 "evaluate_node" (평가)
    - 둘 다 없으면 END (직접 답변 후 종료)
    """
    messages = state.get("messages", [])
    if not messages:
        # 메시지가 없는 예외적인 경우 안전하게 종료
        return "evaluate_node"
    
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

    return "evaluate_node"

        


def route_doc_relevance(state: AgentState) -> str:
    """
    문서 관련성 평가 결과에 따른 라우팅
    - 관련성 높음: "generate" (답변 생성)
    - 관련성 낮음: "analyze_missing_info" (부족한 정보 분석 및 요청)
    """
    relevance_passed = state.get("_doc_relevance_passed", False)
    is_retry = state.get("is_retry", False)
    
    if relevance_passed:
        logger.info("문서 관련성이 높습니다. 답변 생성으로 진행합니다.")
        return "generate"
    
    # [수정] 재시도 상황이면 정보가 부족해도(관련성이 낮아도) 일단 답변 시도
    if is_retry:
        logger.info("🔄 재시도(is_retry) 상황이므로 문서 관련성이 낮아도 강제로 답변을 생성합니다.")
        return "generate"
    
    logger.info("문서 관련성이 낮습니다. 부족한 정보 분석(analyze_missing_info)으로 진행합니다.")
    return "analyze_missing_info"


def create_agent_graph():
    """
    LangGraph 에이전트 그래프 생성 (Self-RAG 구조)
    """
    # Tool 정의 (모든 tool을 LLM에 제공)
    tools = [
        milvus_knowledge_search,  # RAG 검색 tool
        retrieve_qna,             # QnA 검색 tool
    ]
    
    # StateGraph 생성
    workflow = StateGraph(AgentState)
    
    # ToolNode 생성 (Tool 실행 노드)
    tool_node = ToolNode(tools)
    
    # 노드 추가
    workflow.add_node("intent_classifier", intent_classifier_node) # 의도분석
    workflow.add_node("agent", agent_node)  # 질문 분석/도구 호출 결정
    workflow.add_node("tools", tool_node)  # ToolNode: Vector DB 검색
    workflow.add_node("evaluate_node", evaluate_node)  # 검색 결과 관련성 평가
    workflow.add_node("generate", generate_node)  # 답변 생성
    
    # [추가] 응급 상황 노드
    workflow.add_node("emergency_response", emergency_response_node)

    # 엣지 연결
    
    # 0. START -> 의도 분류 (가장 먼저 실행)
    workflow.add_edge(START, "intent_classifier")
    
    # 1. 의도 분류 결과 분기
    workflow.add_conditional_edges(
        "intent_classifier",
        route_intent,
        {
            "agent": "agent",   # 관련 있음 -> 기존 플로우 진입
            "emergency_response": "emergency_response", # 응급 상황 -> 패스트트랙
            END: END # 관련 없음 -> 종료 (이미 응답 생성됨)
        }
    )
    
    # [추가] 응급 상황 플로우 연결
    workflow.add_edge("emergency_response", END)
    
    # 2. Agent -> Tools 결정 (QnA 노드 분기 삭제됨)
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",  # ToolNode: Tool 실행 및 ToolMessage 자동 추가
            "evaluate_node": "evaluate_node", # Tool 호출 없으면 평가 단계로
        }
    )
    
    # 4. Tools 실행 후 -> 다시 Agent로 가서 결과 수집
    workflow.add_edge("tools", "agent")
    
    # 5. evaluate_node -> generate (관련성 높음) 또는 analyze_missing_info (관련성 낮음)
    workflow.add_edge("evaluate_node", "generate")
    
    # 7. generate -> END (바로 종료)
    workflow.add_edge("generate", END)
    
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
