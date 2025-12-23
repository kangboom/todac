"""
Milvus 컬렉션 스키마 정의 및 초기화
"""
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
from app.core.config import settings
from app.core.database import get_milvus_connection
import logging

logger = logging.getLogger(__name__)

# Milvus 컬렉션 이름
MILVUS_COLLECTION_NAME = "knowledge_base"

# 임베딩 차원 (text-embedding-3-small: 1536)
EMBEDDING_DIMENSION = 1536


def create_milvus_collection() -> Collection:
    """
    Milvus 컬렉션 생성 (없으면 생성, 있으면 반환)
    
    스키마:
    - id: int64 (자동 생성)
    - doc_id: varchar (문서 UUID)
    - chunk_index: int64 (청크 인덱스)
    - embedding: float_vector (1536차원)
    - content: varchar (텍스트 내용)
    - filename: varchar (파일명)
    - category: varchar (카테고리)
    - headers: varchar (헤더 정보 JSON 문자열)
    """
    get_milvus_connection()
    
    # 컬렉션이 이미 존재하는지 확인
    if utility.has_collection(MILVUS_COLLECTION_NAME):
        logger.info(f"컬렉션 '{MILVUS_COLLECTION_NAME}'이 이미 존재합니다.")
        collection = Collection(MILVUS_COLLECTION_NAME)
        collection.load()
        return collection
    
    # 필드 스키마 정의
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=36, description="문서 UUID"),
        FieldSchema(name="chunk_index", dtype=DataType.INT64, description="청크 인덱스"),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION, description="임베딩 벡터"),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535, description="텍스트 내용"),
        FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255, description="파일명"),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50, description="카테고리"),
        FieldSchema(name="headers", dtype=DataType.VARCHAR, max_length=2048, description="헤더 정보 JSON 문자열"),
    ]
    
    # 컬렉션 스키마 생성
    schema = CollectionSchema(
        fields=fields,
        description="미숙아 챗봇 지식 베이스"
    )
    
    # 컬렉션 생성
    collection = Collection(
        name=MILVUS_COLLECTION_NAME,
        schema=schema
    )
    
    # 인덱스 생성 (벡터 검색 최적화)
    index_params = {
        "metric_type": "L2",  # 유클리드 거리
        "index_type": "IVF_FLAT",  # IVF_FLAT 인덱스 (소규모 데이터에 적합)
        "params": {"nlist": 1024}
    }
    
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )
    
    # 컬렉션 로드
    collection.load()
    
    logger.info(f"Milvus 컬렉션 '{MILVUS_COLLECTION_NAME}' 생성 완료")
    
    return collection


def get_milvus_collection_safe() -> Collection:
    """
    Milvus 컬렉션 가져오기 (없으면 생성)
    """
    get_milvus_connection()
    
    if utility.has_collection(MILVUS_COLLECTION_NAME):
        collection = Collection(MILVUS_COLLECTION_NAME)
        collection.load()
        return collection
    else:
        return create_milvus_collection()

