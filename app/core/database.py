"""
Postgres & Milvus 연결 세션 관리
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pymilvus import connections, Collection
from app.core.config import settings

# PostgreSQL 설정
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=settings.DEBUG  # SQL 쿼리 로깅 (개발 환경에서만)
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
