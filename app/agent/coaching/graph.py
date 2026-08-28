"""GROW 코칭 V2 그래프 구성과 PostgreSQL 체크포인터."""
from __future__ import annotations

import asyncio

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg_pool import AsyncConnectionPool

from app.agent.coaching.nodes import COACHING_NODES
from app.agent.coaching.state import CoachingState
from app.core.config import settings

_graph = None
_checkpointer = None
_pool = None
_lock = asyncio.Lock()


def build_coaching_graph() -> StateGraph:
    builder = StateGraph(CoachingState)
    for name, node in COACHING_NODES.items():
        builder.add_node(name, node)

    builder.add_edge(START, "safety_gate")
    for prepare in (
        "consent_prepare", "goal_prepare", "goal_confirm_prepare", "reality_prepare",
        "reality_confirm_prepare", "options_prepare", "will_prepare", "confidence_prepare",
        "checkin_prepare", "completion_prepare",
    ):
        builder.add_edge(prepare, "wait_for_user")

    builder.add_edge("apply_goal", "goal_confirm_prepare")
    builder.add_edge("apply_reality", "reality_confirm_prepare")
    builder.add_edge("apply_option", "will_prepare")
    builder.add_edge("apply_checkin", "review")
    builder.add_edge("general_answer", END)
    builder.add_edge("complete_response", END)
    builder.add_edge("escalated_response", END)
    builder.add_edge("emergency_response", END)
    return builder


async def get_coaching_graph():
    global _graph, _checkpointer, _pool
    if _graph is not None:
        return _graph
    async with _lock:
        if _graph is not None:
            return _graph
        _pool = AsyncConnectionPool(
            conninfo=settings.DATABASE_URL,
            max_size=20,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=False,
        )
        await _pool.open()
        _checkpointer = AsyncPostgresSaver(conn=_pool)
        await _checkpointer.setup()
        _graph = build_coaching_graph().compile(checkpointer=_checkpointer)
        return _graph


async def close_coaching_graph() -> None:
    global _graph, _checkpointer, _pool
    async with _lock:
        if _pool is not None:
            await _pool.close()
        _graph = None
        _checkpointer = None
        _pool = None
