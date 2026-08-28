import asyncio
import sys

import pytest

from app.agent.coaching.nodes import common


@pytest.fixture(scope="session")
def event_loop_policy():
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.get_event_loop_policy()


@pytest.fixture(autouse=True)
def disable_external_services(monkeypatch):
    monkeypatch.setattr(common, "get_evaluator_llm", lambda: None)
    monkeypatch.setattr(common, "get_generator_llm", lambda: None)
    monkeypatch.setattr(common, "retrieve_context", lambda query: ("테스트 근거", ["test-source"]))
