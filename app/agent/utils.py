"""
Agent 모듈 유틸리티 함수
ToolMessage 및 LLM 응답 파싱
"""
import json
import logging

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
