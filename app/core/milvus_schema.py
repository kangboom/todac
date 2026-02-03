"""
Milvus 컬렉션 스키마 정의 및 초기화
"""
from pymilvus import (
    DataType,
    Function,
    FunctionType,
)
from app.core.database import get_milvus_client
import logging

logger = logging.getLogger(__name__)

# Milvus 컬렉션 이름
MILVUS_COLLECTION_NAME = "knowledge_base"
OFFICIAL_QNA_COLLECTION_NAME = "official_qna"

# 임베딩 차원 (text-embedding-3-small: 1536)
EMBEDDING_DIMENSION = 1536


def create_milvus_collection():
    """
    [MilvusClient 버전] 지식 베이스 컬렉션 생성
    """
    try:
        client = get_milvus_client()

        # 1. 컬렉션 존재 여부 확인
        if client.has_collection(MILVUS_COLLECTION_NAME):
            logger.info(f"✅ 컬렉션 '{MILVUS_COLLECTION_NAME}'이 이미 존재합니다.")
            client.load_collection(MILVUS_COLLECTION_NAME)
            return

        # 2. 스키마 생성 (MilvusClient 스타일)
        schema = client.create_schema(
            auto_id=True,
            enable_dynamic_field=False,
            description="미숙아 챗봇 지식 베이스"
        )

        analyzer_params_ko = {
            "tokenizer": "standard",
            "filter": [
                "lowercase",
                {
                    "type": "stop",
                    "stop_words": ["은", "는", "이", "가", "를", "을", "의", "에", "와", "과", "도", "만"]
                }
            ]
        }

        # 3. 필드 추가 (add_field 메서드 사용)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=36, description="문서 UUID")
        schema.add_field(field_name="chunk_index", datatype=DataType.INT64, description="청크 인덱스")
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION, description="임베딩 벡터")
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535, description="텍스트 내용", enable_match=True, enable_analyzer=True, analyzer_params=analyzer_params_ko)
        schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=255, description="파일명")
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=50, description="카테고리")
        schema.add_field(field_name="headers", datatype=DataType.VARCHAR, max_length=2048, description="헤더 정보 JSON 문자열")
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # BM25 Function: content 필드를 sparse vector로 변환
        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["content"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        # 4. 인덱스 설정 (prepare_index_params 사용)
        index_params = client.prepare_index_params()

        # 벡터 인덱스 (Dense)
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",  # 소규모 데이터에 적합 (데이터 커지면 IVF_SQ8이나 HNSW 추천)
            metric_type="L2",       # 유클리드 거리
            params={"nlist": 1024}
        )

        # Sparse Index 
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25", 
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": 1.2,
                "bm25_b": 0.75
            }
        )

        # 5. 컬렉션 생성 (스키마 + 인덱스 한번에)
        client.create_collection(
            collection_name=MILVUS_COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

        # 6. 로드 (검색 준비)
        client.load_collection(MILVUS_COLLECTION_NAME)
        
        logger.info(f"🎉 Milvus 컬렉션 '{MILVUS_COLLECTION_NAME}' 생성 및 로드 완료")

    except Exception as e:
        logger.error(f"❌ 컬렉션 생성 실패: {str(e)}")
        raise e


def create_qna_collection():
    """
    공식 문서 스타일(MilvusClient)로 QnA 컬렉션 생성
    """
    try:
        client = get_milvus_client()

        if client.has_collection(OFFICIAL_QNA_COLLECTION_NAME):
            client.drop_collection(OFFICIAL_QNA_COLLECTION_NAME)


        # 2. 스키마 생성 (Auto ID, Analyzer 설정)
        schema = client.create_schema(
            auto_id=True, 
            enable_dynamic_field=False, 
            description="공식 QnA 데이터베이스 (Hybrid)"
        )

        # 3. 필드 추가 (add_field 메서드 사용)
        # 한국어 분석기 설정
        analyzer_params_ko = {
            "tokenizer": "standard",
            "filter": [
                "lowercase",
                {
                    "type": "stop",
                    "stop_words": ["은", "는", "이", "가", "를", "을", "의", "에", "와", "과", "도", "만"]
                }
            ]
        }

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="qna_id", datatype=DataType.INT64)
        schema.add_field(field_name="question", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field(field_name="answer", datatype=DataType.VARCHAR, max_length=65535)
        # question + answer 통합 필드 (BM25 검색용)
        schema.add_field(
            field_name="question_answer", 
            datatype=DataType.VARCHAR, 
            max_length=65535,
            enable_match=True,
            enable_analyzer=True, 
            analyzer_params=analyzer_params_ko
        )
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=1536)
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR) 

        # 4. Function 정의 및 스키마에 추가 - question_answer 필드 사용
        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["question_answer"],  # question 대신 question_answer 사용
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        # 5. 인덱스 설정 (prepare_index_params 사용)
        index_params = client.prepare_index_params()

        # Dense Index
        index_params.add_index(
            field_name="embedding", 
            index_type="IVF_FLAT",
            metric_type="L2", # 또는 IP
            params={"nlist": 128}
        )

        # Sparse Index (중요: metric_type="BM25"를 쓰면 SDK가 알아서 IP로 처리해줌)
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25", # High-level SDK에서는 "BM25"라고 적어도 됨 (자동변환)
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": 1.2,
                "bm25_b": 0.75
            }
        )

        client.create_collection(
            collection_name=OFFICIAL_QNA_COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

        client.load_collection(OFFICIAL_QNA_COLLECTION_NAME)
        logger.info(f"🎉 Milvus 컬렉션 '{OFFICIAL_QNA_COLLECTION_NAME}' 생성 완료")
        
    except Exception as e:
        logger.error(f"❌ 컬렉션 생성 실패: {str(e)}")
        raise e


