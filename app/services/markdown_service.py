"""
Markdown 보정 서비스 (LLM 활용)
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.agent.prompts import MARKDOWN_CLEANUP_PROMPT
from app.core.config import settings
import logging

from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache()
def get_cleanup_model():
    """Markdown 보정용 LLM 클라이언트 싱글톤 (LRU Cache 사용)"""
    if settings.OPENAI_API_KEY:
        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL_GENERATION,
            temperature=0.1,
            max_tokens=4000
        )
    return None


def cleanup_markdown_with_llm(markdown_text: str, filename: str = None) -> str:
    """
    LLM을 사용하여 Markdown 문서 보정
    
    Args:
        markdown_text: 보정할 Markdown 텍스트
        filename: 파일명 (로깅용)
    
    Returns:
        보정된 Markdown 텍스트
    """
    if not markdown_text or not markdown_text.strip():
        return markdown_text
    
    cleanup_model = get_cleanup_model()
    if not cleanup_model:
        logger.warning("LLM 클라이언트가 없어 Markdown 보정을 건너뜁니다.")
        return markdown_text
    
    try:
        logger.info(f"Markdown 보정 시작: 파일={filename}, 길이={len(markdown_text)}")
        
        # 프롬프트 생성
        messages = [
            SystemMessage(content=MARKDOWN_CLEANUP_PROMPT),
            HumanMessage(content=f"보정할 Markdown 문서:\n\n{markdown_text}")
        ]
        
        # LLM 호출
        response = cleanup_model.invoke(messages)
        cleaned_text = response.content.strip()
        
        logger.info(f"Markdown 보정 완료: 파일={filename}, 원본 길이={len(markdown_text)}, 보정 후 길이={len(cleaned_text)}")
        
        return cleaned_text
        
    except Exception as e:
        logger.error(f"Markdown 보정 실패: {str(e)}, 원본 텍스트 반환")
        # 보정 실패 시 원본 반환
        return markdown_text

