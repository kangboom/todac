"""
환경변수 로드 (.env)
"""
from pydantic_settings import BaseSettings
from typing import Optional
from langchain_openai import OpenAIEmbeddings
import logging

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # 데이터베이스 설정
    DATABASE_URL: str = "postgresql://postgres:postgres@postgres:5432/todac_db"
    
    # Milvus 설정
    MILVUS_HOST: str = "milvus"
    MILVUS_PORT: int = 19530
    
    @property
    def MILVUS_URI(self) -> str:
        """Milvus URI 생성 (MilvusClient용)"""
        return f"http://{self.MILVUS_HOST}:{self.MILVUS_PORT}"
    
    # JWT 설정
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # OpenAI 설정
    OPENAI_API_KEY: Optional[str] = None
    
    # PDF 파서 설정
    PDF_PARSER: str = "pymupdf"  # "pymupdf", "llamaparse", 또는 "docling"
    
    # LlamaParse 설정
    LLAMAPARSE_API_KEY: Optional[str] = None  # LlamaParse API 키 (무료 사용량 제한 있음)
    LLAMAPARSE_OPENAI_MODEL: str = "openai-gpt-4o-mini" # LlamaParse 사용할 OpenAI 모델
    # OpenAI 모델 설정 (노드별)
    OPENAI_MODEL_INTENT: str = "gpt-4o-mini"  # 의도 분류 모델
    OPENAI_MODEL_REWRITE: str = "gpt-4o-mini"  # 쿼리 재작성 모델
    OPENAI_MODEL_GENERATION: str = "gpt-4o-mini"  # 답변 생성 모델
    OPENAI_MODEL_EMBEDDING: str = "text-embedding-3-small"  # 임베딩 모델
    
    # RAG 설정
    MAX_RAG_RETRIEVAL_ATTEMPTS: int = 3  # 최대 RAG 검색 시도 횟수
    MIN_RAG_SCORE_THRESHOLD: float = 0.8  # 최소 RAG 스코어 임계값
    
    # 환경 설정
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    # S3 설정
    S3_BUCKET_NAME: str = "my-rag-bucket"
    S3_REGION: str = "ap-northeast-2"  # 서울 리전
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    S3_ENDPOINT_URL: Optional[str] = None  # MinIO 등 로컬 S3 호환 서비스 사용 시


settings = Settings()

# OpenAIEmbeddings 싱글톤 인스턴스
_embeddings: Optional[OpenAIEmbeddings] = None


def get_embeddings() -> OpenAIEmbeddings:
    """
    OpenAIEmbeddings 싱글톤 인스턴스 가져오기
    """
    global _embeddings
    
    if _embeddings is None:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        try:
            logger.info(f"OpenAIEmbeddings 초기화: model={settings.OPENAI_MODEL_EMBEDDING}")
            _embeddings = OpenAIEmbeddings(
                model=settings.OPENAI_MODEL_EMBEDDING,
                openai_api_key=settings.OPENAI_API_KEY,
                chunk_size=200,
            )
            logger.info("OpenAIEmbeddings 초기화 완료")
        except Exception as e:
            logger.error(f"OpenAIEmbeddings 초기화 실패: {e}")
            raise
    
    return _embeddings


def reset_embeddings():
    """
    OpenAIEmbeddings 인스턴스 리셋 (테스트 또는 재초기화 시 사용)
    """
    global _embeddings
    _embeddings = None
