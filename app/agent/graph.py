"""
Workflow 정의 (Coaching Agent - StateGraph, Edge 연결)
"""
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.prebuilt import ToolNode
from psycopg_pool import AsyncConnectionPool
from app.agent.state import AgentState
from app.agent.nodes import (
    intent_classifier_node,
    emergency_response_node,
    goal_setter_node,
    goal_evaluator_node,
    coach_agent_node,
    coaching_evaluator_node,
    closing_node,
)
from app.agent.tools import milvus_knowledge_search, retrieve_qna
from app.core.config import settings
from langchain_core.messages import AIMessage
import logging

logger = logging.getLogger(__name__)

# 코칭 에이전트에서 사용하는 도구 목록
coaching_tools = [milvus_knowledge_search, retrieve_qna]


def route_intent(state: AgentState) -> str:
    """
    의도 분류 결과에 따른 라우팅
    - emergency: 응급 상황 패스트트랙
    - irrelevant: 단순 응답 후 종료 (이미 intent_classifier에서 응답 생성됨)
    - relevant: 코칭 플로우 진입 (Goal Setter)
    """
    intent = state.get("_intent", "relevant")
    
    if intent == "emergency":
        logger.info("🚨 응급 상황 감지 -> Emergency Fast-Track 진입")
        return "emergency_response"

    if intent == "irrelevant":
        logger.info("🚫 질문이 아기 돌봄과 관련이 없습니다 -> 단순 응답 후 종료")
        return END
    
    logger.info("✅ 질문이 관련성이 있습니다 -> Goal Setter 노드 진입")
    return "goal_setter"


def route_goal_evaluator(state: AgentState) -> str:
    """
    Goal Evaluator 결과에 따른 라우팅
    - approved: coach_agent로 진행 (코칭 시작)
    - modify: goal_setter로 복귀 (사용자 피드백 반영하여 재설정)
    """
    goal_approved = state.get("_goal_approved", True)
    
    if goal_approved:
        logger.info("✅ 목표 승인 → Coach Agent 노드 진입")
        return "coach_agent"
    
    logger.info("✏️ 목표 수정 요청 → Goal Setter 노드 재진입")
    return "goal_setter"


def route_coach_agent(state: AgentState) -> str:
    """
    Coach Agent 출력에 따른 라우팅
    - tool_calls가 있으면 → tool_node (검색 도구 실행)
    - tool_calls가 없으면 → evaluator (응답 완료, interrupt 후 사용자 대기)
    """
    messages = state.get("messages", [])
    
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
            logger.info("🔧 Coach Agent → ToolNode (검색 도구 실행)")
            return "tool_node"
    
    logger.info("✅ Coach Agent → Evaluator (응답 완료)")
    return "evaluator"


def route_evaluator(state: AgentState) -> str:
    """
    Evaluator 결과에 따른 라우팅
    - completed/paused → closing (마무리)
    - 그 외 (다음 단계, 재시도, 잡담) → coach_agent (루프)
    """
    goal_status = state.get("goal_status", "in_progress")
    
    if goal_status in ("completed", "paused"):
        logger.info(f"🏁 코칭 종료 -> Closing 노드 (status={goal_status})")
        return "closing"
    
    logger.info("🔄 코칭 계속 -> Coach Agent 노드 (루프)")
    return "coach_agent"


def route_goal_setter(state: AgentState) -> str:
    """
    Goal Setter 출력에 따른 라우팅
    - tool_calls가 있으면 → goal_setter_tool (검색 도구 실행)
    - tool_calls가 없으면 → goal_evaluator (목표 수립 완료)
    """
    messages = state.get("messages", [])
    
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
            logger.info("🔧 Goal Setter → ToolNode (검색 도구 실행)")
            return "goal_setter_tool"
            
    logger.info("✅ Goal Setter → Goal Evaluator (목표 수립 완료)")
    return "goal_evaluator"


def create_coaching_graph_builder() -> StateGraph:
    """
    코칭 에이전트 StateGraph 빌더 생성
    
    그래프 구조:
    START → intent_classifier
      ├─ emergency → emergency_response → END
      ├─ irrelevant → END
      └─ relevant → goal_setter
                      ├─ tool_calls → goal_setter_tool → goal_setter (루프)
                      └─ 완료 → [INTERRUPT] → goal_evaluator
                                                  ├─ approved → coach_agent
                                                  │               ├─ tool_calls → tool_node → coach_agent (루프)
                                                  │               └─ 응답완료 → [INTERRUPT] → evaluator
                                                  │                                           ├─ completed/paused → closing → END
                                                  │                                           └─ 계속 → coach_agent (루프)
                                                  └─ modify → goal_setter (루프, 피드백 반영하여 재설정)
    """
    workflow = StateGraph(AgentState)
    
    # ===== 노드 등록 =====
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("emergency_response", emergency_response_node)
    workflow.add_node("goal_setter", goal_setter_node)
    workflow.add_node("goal_setter_tool", ToolNode(coaching_tools)) # Goal Setter 전용 도구 노드
    workflow.add_node("goal_evaluator", goal_evaluator_node)
    workflow.add_node("coach_agent", coach_agent_node)
    workflow.add_node("tool_node", ToolNode(coaching_tools))
    workflow.add_node("evaluator", coaching_evaluator_node)
    workflow.add_node("closing", closing_node)
    
    # ===== 엣지 연결 =====
    
    # 0. START -> 의도 분류
    workflow.add_edge(START, "intent_classifier")
    
    # 1. 의도 분류 결과 분기
    workflow.add_conditional_edges(
        "intent_classifier",
        route_intent,
        {
            "goal_setter": "goal_setter",
            "emergency_response": "emergency_response",
            END: END
        }
    )
    
    # 2. 응급 상황 -> END
    workflow.add_edge("emergency_response", END)
    
    # 3. Goal Setter -> 조건부 분기 (Tool 사용 or 완료)
    workflow.add_conditional_edges(
        "goal_setter",
        route_goal_setter,
        {
            "goal_setter_tool": "goal_setter_tool",
            "goal_evaluator": "goal_evaluator"
        }
    )
    
    # 3-1. Goal Setter Tool -> Goal Setter (결과 반환 후 루프)
    workflow.add_edge("goal_setter_tool", "goal_setter")
    
    # 4. Goal Evaluator -> 조건부 분기
    #    - approved → coach_agent (코칭 시작)
    #    - modify → goal_setter (피드백 반영 재설정, 루프)
    workflow.add_conditional_edges(
        "goal_evaluator",
        route_goal_evaluator,
        {
            "coach_agent": "coach_agent",
            "goal_setter": "goal_setter"
        }
    )
    
    # 5. Coach Agent -> 조건부 분기
    #    - tool_calls 있음 → tool_node (검색 도구 실행)
    #    - tool_calls 없음 → evaluator (응답 완료, interrupt 후 사용자 대기)
    workflow.add_conditional_edges(
        "coach_agent",
        route_coach_agent,
        {
            "tool_node": "tool_node",
            "evaluator": "evaluator"
        }
    )
    
    # 6. ToolNode -> Coach Agent (검색 결과 반환 후 재호출)
    workflow.add_edge("tool_node", "coach_agent")
    
    # 7. Evaluator -> 조건부 분기
    workflow.add_conditional_edges(
        "evaluator",
        route_evaluator,
        {
            "coach_agent": "coach_agent",
            "closing": "closing"
        }
    )
    
    # 8. Closing -> END
    workflow.add_edge("closing", END)
    
    return workflow


# 전역 그래프 인스턴스 (한 번만 생성)
_agent_graph = None
_checkpointer = None


async def get_agent_graph():
    """
    에이전트 그래프 인스턴스 가져오기 (싱글톤, async)
    AsyncPostgresSaver를 체크포인터로 사용하며,
    2개의 interrupt 포인트에서 HITL(Human-in-the-Loop)을 구현합니다.
    
    interrupt 위치:
    1. goal_setter → goal_evaluator 사이 (목표/계획 사용자 승인 대기)
    2. coach_agent → evaluator 사이 (코칭 가이드 후 사용자 응답 대기)
    """
    global _agent_graph, _checkpointer
    
    if _agent_graph is None:
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
            interrupt_before=["goal_evaluator", "evaluator"]  # 2개의 HITL 포인트
        )
        
        logger.info("✅ 코칭 에이전트 그래프 컴파일 완료 (interrupt_before=['goal_evaluator', 'evaluator'])")
    
    return _agent_graph
