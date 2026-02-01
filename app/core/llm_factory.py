from functools import lru_cache
from langchain_openai import ChatOpenAI
from app.core.config import settings

@lru_cache(maxsize=1)
def get_generator_llm() -> ChatOpenAI | None:
    """
    대화 생성 및 질문 재작성용 (창의성 필요)
    Temperature: 0.7
    """
    if not settings.OPENAI_API_KEY:
        return None
        
    return ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_MODEL_GENERATION,
        temperature=0.7,
        max_tokens=1000
    )

@lru_cache(maxsize=1)
def get_evaluator_llm() -> ChatOpenAI | None:
    """
    의도 분류, 문서 평가, JSON 추출용 (정확성 필요)
    Temperature: 0.1
    """
    if not settings.OPENAI_API_KEY:
        return None
        
    return ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_MODEL_GENERATION,
        temperature=0.1,
        max_tokens=600
    )
