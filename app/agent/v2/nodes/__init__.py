"""GROW 코칭 노드 모음.

노드 구현은 책임별 모듈에 두고, 그래프에는 단일 레지스트리로 제공한다.
기존 ``app.agent.v2.nodes.<node_name>`` 접근도 유지한다.
"""
from __future__ import annotations

from app.agent.v2.nodes.action import ACTION_NODES
from app.agent.v2.nodes.goal import GOAL_NODES
from app.agent.v2.nodes.reality import REALITY_NODES
from app.agent.v2.nodes.review import REVIEW_NODES
from app.agent.v2.nodes.routing import ROUTING_NODES

COACHING_NODES = {
    **ROUTING_NODES,
    **GOAL_NODES,
    **REALITY_NODES,
    **ACTION_NODES,
    **REVIEW_NODES,
}

# 함수 단위로 import하던 코드와 테스트의 공개 경로를 보존한다.
EXPORTED_NODE_FUNCTIONS = {node.__name__: node for node in COACHING_NODES.values()}
globals().update(EXPORTED_NODE_FUNCTIONS)

__all__ = ["COACHING_NODES", *EXPORTED_NODE_FUNCTIONS]
