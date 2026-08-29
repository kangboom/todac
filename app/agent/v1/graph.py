"""
Workflow 정의 (Coaching Agent - StateGraph, Edge 연결)
"""
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from app.agent.state import AgentState
from app.agent.v1.nodes import (
    intent_classifier_node,
    emergency_response_node,
    ask_situation_node,
    goal_options_node,
    goal_selector_node,
    research_agent_node,
    evaluate_docs_node,
    grow_response_node
)
from app.core.config import settings
import asyncio
import logging

logger = logging.getLogger(__name__)


def route_intent(state: AgentState) -> str:
    """
    의도 분류 결과에 따른 라우팅
    - emergency: 응급 상황 패스트트랙
    - irrelevant: 단순 응답 후 종료 (이미 intent_classifier에서 응답 생성됨)
    - relevant: 코칭 플로우 진입 (Ask Situation)
    """
    intent = state.get("_intent", "relevant")
    
    if intent == "emergency":
        logger.info("🚨 응급 상황 감지 -> Emergency Fast-Track 진입")
        return "emergency_response"

    if intent == "irrelevant":
        logger.info("🚫 질문이 아기 돌봄과 관련이 없습니다 -> 단순 응답 후 종료")
        return END
    
    logger.info("✅ 질문이 관련성이 있습니다 -> Ask Situation 노드 진입")
    return "ask_situation"


def route_goal_selector(state: AgentState) -> str:
    """
    Goal Selector 결과에 따른 라우팅
    - _goal_valid == False: 관련 없는 응답 → self-loop (다시 목표 선택 대기)
    - _goal_valid == True: 유효한 목표 → Research Agent 진입
    """
    if state.get("_goal_valid") == False:
        logger.info("🔄 목표 미설정 → goal_selector self-loop")
        return "goal_selector"
    
    logger.info("✅ 목표 설정 완료 → research_agent 진입")
    return "research_agent"


def create_coaching_graph_builder() -> StateGraph:
    """
    코칭 에이전트 StateGraph 빌더 생성
    """
    workflow = StateGraph(AgentState)
    
    # ===== 노드 등록 =====
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("emergency_response", emergency_response_node)
    workflow.add_node("ask_situation", ask_situation_node)
    workflow.add_node("goal_options", goal_options_node)
    workflow.add_node("goal_selector", goal_selector_node)
    workflow.add_node("research_agent", research_agent_node)
    workflow.add_node("evaluate_docs", evaluate_docs_node)
    workflow.add_node("response_node", grow_response_node)
    
    # ===== 엣지 연결 =====
    
    # 0. START -> 의도 분류
    workflow.add_edge(START, "intent_classifier")
    
    # 1. 의도 분류 결과 분기
    workflow.add_conditional_edges(
        "intent_classifier",
        route_intent,
        {
            "ask_situation": "ask_situation",
            "emergency_response": "emergency_response",
            END: END
        }
    )
    
    # 2. 응급 상황 -> END
    workflow.add_edge("emergency_response", END)
    
    # 3. Ask Situation -> Goal Options (interrupt_before로 1차 멈춤)
    workflow.add_edge("ask_situation", "goal_options")
    
    # 4. Goal Options -> Goal Selector (interrupt_before로 2차 멈춤)
    workflow.add_edge("goal_options", "goal_selector")
    
    # 5. Goal Selector -> 조건부 분기 (관련 없는 응답이면 self-loop)
    workflow.add_conditional_edges(
        "goal_selector",
        route_goal_selector,
        {
            "goal_selector": "goal_selector",
            "research_agent": "research_agent"
        }
    )
    
    # 6. Research Agent -> Evaluate Docs
    workflow.add_edge("research_agent", "evaluate_docs")
    
    # 7. Evaluate Docs -> Response Node
    workflow.add_edge("evaluate_docs", "response_node")
    
    # 8. Response Node -> END
    workflow.add_edge("response_node", END)
    
    return workflow


# 전역 그래프 인스턴스 (한 번만 생성)
_agent_graph = None
_checkpointer = None
_graph_lock = asyncio.Lock()


async def get_agent_graph():
    """
    에이전트 그래프 인스턴스 가져오기 (싱글톤, async + Lock)
    
    interrupt 위치:
    - goal_options 노드 진입 전: Ask Situation이 질문을 던진 후, 사용자의 상황 답변을 받기 위해 멈춤.
    - goal_selector 노드 진입 전: Goal Options가 선택지를 던진 후, 사용자의 목표 선택을 받기 위해 멈춤.
    """
    global _agent_graph, _checkpointer
    
    # Fast path: 이미 초기화된 경우 Lock 없이 바로 반환
    if _agent_graph is not None:
        return _agent_graph
    
    # 초기화 시에만 Lock 획득 (동시 초기화 방지)
    async with _graph_lock:
        # Double-check: Lock 대기 중 다른 코루틴이 이미 초기화했을 수 있음
        if _agent_graph is not None:
            return _agent_graph
        
        db_uri = settings.DATABASE_URL
        
        pool = AsyncConnectionPool(
            conninfo=db_uri,
            max_size=20,
            kwargs={"autocommit": True, "prepare_threshold": 0}
        )
        await pool.open()
        
        _checkpointer = AsyncPostgresSaver(conn=pool)
        await _checkpointer.setup()
        
        logger.info("✅ AsyncPostgresSaver 체크포인터 초기화 완료")
        
        builder = create_coaching_graph_builder()
        
        _agent_graph = builder.compile(
            checkpointer=_checkpointer,
            interrupt_before=["goal_options", "goal_selector"]
        )
        
        logger.info("✅ 코칭 그래프 컴파일 완료 (interrupt_before=['goal_options', 'goal_selector'])")
    
    return _agent_graph
