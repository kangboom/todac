"""
Agent 모듈 유틸리티 함수
ToolMessage 및 LLM 응답 파싱
"""
import json
import logging
from typing import List, Callable, Awaitable
from langchain_core.messages import BaseMessage
import time
from functools import wraps
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


def parse_tool_result(content: str | list) -> list:
    """
    ToolMessage의 content를 파싱하여 리스트로 반환
    
    Args:
        content: ToolMessage의 content (str 또는 list)
        
    Returns:
        파싱된 리스트 (실패 시 빈 리스트)
    """
    if isinstance(content, list):
        return content
    if isinstance(content, str):
        try:
            # JSON 문자열 파싱
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
            return []
        except json.JSONDecodeError:
            return []
    return []


def parse_json_from_response(text: str) -> dict:
    """
    LLM 응답 텍스트에서 JSON을 추출하여 파싱
    
    Args:
        text: LLM 응답 텍스트 (마크다운 코드 블록 포함 가능)
        
    Returns:
        파싱된 딕셔너리 (실패 시 빈 딕셔너리)
    """
    try:
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        return json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"JSON 파싱 실패: {text[:50]}...")
        return {}
    except Exception as e:
        logger.error(f"JSON 추출 중 오류: {str(e)}")
        return {}


def log_message_history(messages: List[BaseMessage], max_content_length: int = 100, context: str = ""):
    """
    메시지 히스토리를 요약하여 로깅
    
    Args:
        messages: 로깅할 메시지 리스트
        max_content_length: 각 메시지 내용의 최대 표시 길이 (기본값: 100)
        context: 로그에 추가할 컨텍스트 문자열 (예: "generate_node", "intent_classifier")
    """
    if not messages:
        context_str = f" [{context}]" if context else ""
        logger.info(f"📜 히스토리 없음 (첫 대화){context_str}")
        return
    
    history_summary = []
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        content = getattr(msg, 'content', '')
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        history_summary.append(f"[{i+1}] {msg_type}: {content}")
    
    context_str = f" [{context}]" if context else ""
    logger.info(f"📜 최근 히스토리 ({len(messages)}개){context_str}:\n" + "\n".join(history_summary))

def track_node_execution_time(node_name: str):
    """
    노드 실행 시간을 추적하는 데코레이터
    
    Args:
        node_name: 노드 이름 (로깅용)
    
    Usage:
        @track_node_execution_time("intent_classifier")
        async def intent_classifier_node(state: AgentState) -> AgentState:
            ...
    """
    def decorator(func: Callable[[AgentState], Awaitable[AgentState]]):
        @wraps(func)
        async def wrapper(state: AgentState) -> AgentState:
            start_time = time.time()
            try:
                result = await func(state)
                elapsed_time = time.time() - start_time
                logger.info(f"====== ⏱️ [{node_name}] 실행 시간: {elapsed_time:.3f}초 ({elapsed_time*1000:.2f}ms) ⏱️ =======")
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(f"====== ⏱️ [{node_name}] 실행 시간: {elapsed_time:.3f}초 (에러 발생: {str(e)}) ⏱️ =======")
                raise
        return wrapper
    return decorator