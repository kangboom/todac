"""
Postgres & Milvus 연결 세션 관리
"""
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pymilvus import connections, Collection, MilvusClient
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# PostgreSQL 설정
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False  # SQL 쿼리 로깅 비활성화 (너무 많은 로그 방지)
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """데이터베이스 세션 의존성"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Milvus 연결
def get_milvus_connection():
    """Milvus 연결 가져오기"""
    if not connections.has_connection("default"):
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT
        )
    return connections.get_connection_addr("default")


def get_milvus_collection(collection_name: str) -> Collection:
    """Milvus 컬렉션 가져오기"""
    get_milvus_connection()
    collection = Collection(collection_name)
    collection.load()
    return collection


# MilvusClient 싱글톤 인스턴스
_milvus_client: Optional[MilvusClient] = None


def get_milvus_client() -> MilvusClient:
    """
    MilvusClient 싱글톤 인스턴스 가져오기 (High-level API)
    """
    global _milvus_client
    
    if _milvus_client is None:
        try:
            uri = settings.MILVUS_URI
            logger.info(f"MilvusClient 연결 시도: {uri}")
            _milvus_client = MilvusClient(uri=uri)
            logger.info("MilvusClient 연결 성공")
        except Exception as e:
            logger.error(f"MilvusClient 연결 실패: {e}")
            raise
    
    return _milvus_client


def reset_milvus_client():
    """
    MilvusClient 연결 리셋 (테스트 또는 재연결 시 사용)
    """
    global _milvus_client
    if _milvus_client is not None:
        try:
            _milvus_client.close()
        except Exception as e:
            logger.warning(f"MilvusClient 종료 중 오류: {e}")
    _milvus_client = None
