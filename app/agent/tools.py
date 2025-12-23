"""
Milvus 검색 도구 (Hybrid Search 구현)
"""
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from pymilvus import Collection
from app.core.database import get_milvus_collection
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# LangChain OpenAI Embeddings 클라이언트
embeddings = OpenAIEmbeddings(
    api_key=settings.OPENAI_API_KEY,
    model=settings.OPENAI_MODEL_EMBEDDING
) if settings.OPENAI_API_KEY else None

# Milvus 컬렉션 이름
MILVUS_COLLECTION_NAME = "knowledge_base"


def get_embedding(text: str) -> List[float]:
    """텍스트를 임베딩 모델로 임베딩 (환경 변수에서 모델 가져오기)"""
    if not embeddings:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
    
    try:
        embedding = embeddings.embed_query(text)
        return embedding
    except Exception as e:
        logger.error(f"임베딩 생성 실패: {str(e)}")
        raise


@tool
def milvus_knowledge_search(
    query: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    미숙아 관련 지식베이스에서 전문 의학 정보를 검색합니다.
    
    이 tool은 다음 상황에서 반드시 사용하세요:
    - 미숙아 관련 의학 정보, 증상, 질병에 대한 질문
    - 돌봄 방법, 수유, 수면, 발달 관련 질문
    - 예방접종, 일상 관리, 주의사항에 대한 질문
    - 특정 증상이나 상황에 대한 정보가 필요할 때
    - "무엇을 해야 하나요?", "어떻게 해야 하나요?", "왜 그런가요?" 같은 질문
    
    구체적인 사용 예시:
    - "지속적으로 무호흡과 서맥이 발생하거나 증상이 심해지는 경우는 어떻게 하면 좋아?"
    - "미숙아 수유 방법"
    - "호흡곤란 증상"
    - "서맥이 발생하는 이유"
    - "무호흡이 지속될 때 대처 방법"
    - "미숙아 발달 단계"
    
    응급 상황(즉시 119 신고가 필요한 경우)이 아닌 모든 의학 정보 질문에 이 tool을 사용하세요.
    
    Args:
        query: 검색할 질문이나 키워드 (예: "무호흡 서맥 대처", "미숙아 수유 방법", "호흡곤란 증상")
        top_k: 반환할 문서 개수 (기본값: 5)
    
    Returns:
        검색된 문서 리스트 (doc_id, content, score, filename, category 포함)
    """
    try:
        logger.info(f"=== Milvus 검색 시작 ===")
        logger.info(f"검색 질문: {query}")
        logger.info(f"상위 K개: {top_k}")
        
        # 컬렉션 가져오기
        collection = get_milvus_collection(MILVUS_COLLECTION_NAME)
        logger.info(f"컬렉션 '{MILVUS_COLLECTION_NAME}' 가져오기 완료")
        
        # 컬렉션 상태 확인
        collection.load()
        logger.info("컬렉션 로드 완료")
        
        # 데이터 개수 확인
        num_entities = collection.num_entities
        logger.info(f"컬렉션 엔티티 수: {num_entities}")
        
        if num_entities == 0:
            logger.warning("⚠️ Milvus 컬렉션에 데이터가 없습니다. 문서를 먼저 업로드해주세요.")
            return []
        
        # 인덱스 확인
        indexes = collection.indexes
        logger.info(f"컬렉션 인덱스 개수: {len(indexes)}")
        if indexes:
            for idx in indexes:
                logger.info(f"  - 인덱스 필드: {idx.field_name}, 타입: {idx.params}")
        else:
            logger.warning("⚠️ Milvus 컬렉션에 인덱스가 없습니다. 인덱스가 없으면 검색이 실패할 수 있습니다.")
        
        # 질문 임베딩
        logger.info("질문 임베딩 생성 중...")
        query_embedding = get_embedding(query)
        logger.info(f"질문 임베딩 생성 완료: 차원={len(query_embedding)}")
        
        # 검색 파라미터 (데이터가 적을 때 nprobe 조정)
        # nprobe는 nlist보다 작아야 함 (기본 nlist=1024)
        nprobe = min(10, max(1, num_entities // 100 + 1))
        search_params = {
            "metric_type": "L2",  # 유클리드 거리
            "params": {"nprobe": nprobe}
        }
        logger.info(f"검색 파라미터: {search_params}")
        
        # 벡터 검색 수행 (카테고리 필터 없이 전체 검색)
        logger.info(f"Milvus 벡터 검색 실행 중...")
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",  # 임베딩 필드명
            param=search_params,
            limit=top_k,
            output_fields=["doc_id", "content", "filename", "category", "chunk_index", "headers"]
        )
        
        logger.info(f"검색 결과 수신: {len(results)}개 쿼리 결과")
        if results and len(results) > 0:
            logger.info(f"첫 번째 쿼리 결과 수: {len(results[0])}개")
        else:
            logger.warning("⚠️ 검색 결과가 비어있습니다.")
        
        # 결과 파싱
        retrieved_docs = []
        if results and len(results) > 0:
            for idx, hit in enumerate(results[0]):
                doc = {
                    "doc_id": hit.entity.get("doc_id"),
                    "content": hit.entity.get("content", ""),
                    "filename": hit.entity.get("filename", ""),
                    "category": hit.entity.get("category", ""),
                    "chunk_index": hit.entity.get("chunk_index", 0),
                    "headers": hit.entity.get("headers", "{}"),
                    "score": hit.distance,  # 거리 (낮을수록 유사)
                }
                retrieved_docs.append(doc)
                logger.info(
                    f"  [{idx+1}] doc_id={doc['doc_id']}, "
                    f"chunk_index={doc['chunk_index']}, "
                    f"score={doc['score']:.4f}, "
                    f"filename={doc['filename']}, "
                    f"category={doc['category']}, "
                    f"content_length={len(doc['content'])}"
                )
        
        logger.info(f"=== Milvus 검색 완료: {len(retrieved_docs)}개 문서 검색됨 ===")
        return retrieved_docs
        
    except Exception as e:
        logger.error(f"❌ Milvus 검색 실패: {str(e)}", exc_info=True)
        import traceback
        logger.error(f"상세 에러:\n{traceback.format_exc()}")
        # 에러 발생 시 빈 리스트 반환
        return []


@tool
def emergency_protocol_handler(
    symptoms: str,
    urgency_level: str = "high"
) -> str:
    """
    응급 상황을 처리합니다. **현재 진행 중인** 응급 증상이 있을 때만 이 tool을 호출하세요.
    
    ⚠️ 중요: 다음 경우에는 이 tool을 사용하지 마세요:
    - "만약 ~하는 경우", "~하면", "~할 때" 같은 가정형 질문
    - 일반적인 정보나 지식을 묻는 질문
    - 과거에 발생했던 증상에 대한 질문
    - 예방이나 대처 방법을 묻는 질문
    
    ✅ 이 tool을 사용해야 하는 경우 (현재 진행 중인 응급 상황):
    - "아기가 지금 무호흡을 하고 있어요"
    - "현재 서맥이 발생하고 있습니다"
    - "지금 호흡이 멈췄어요"
    - "아기가 지금 경련을 하고 있어요"
    - "현재 청색증이 보여요"
    - "지금 의식이 없어요"
    
    ❌ 이 tool을 사용하지 말아야 하는 경우:
    - "만약 무호흡이 발생하면 어떻게 하나요?" (가정형 질문)
    - "무호흡이 발생하는 경우는?" (정보 질문)
    - "지속적으로 무호흡이 발생하는 경우는 어떻게 하면 좋아?" (정보 질문)
    - "서맥이 발생하는 이유는?" (정보 질문)
    
    응급 증상 목록 (현재 진행 중인 경우만):
    - 호흡곤란, 무호흡, 청색증 (입술이나 손발이 파랗게 변함)
    - 경련, 의식 저하, 반응 없음
    - 심한 호흡음, 기침, 호흡이 멈춤
    - 체온 이상 (고열 38.5도 이상 또는 저체온 36도 미만)
    - 심한 탈수 증상 (소변량 급격히 감소, 눈물 없음, 입술 건조)
    - 심한 구토나 설사로 인한 탈수
    - 출혈이 멈추지 않음
    
    Args:
        symptoms: **현재 관찰되고 있는** 응급 증상에 대한 상세 설명
        urgency_level: 응급도 수준 ("high", "critical") - 기본값: "high"
    
    Returns:
        응급 대응 프로토콜 메시지
    """
    from app.agent.prompts import EMERGENCY_PROTOCOL
    logger.info(f"🚨 응급 프로토콜 호출: 증상={symptoms}, 응급도={urgency_level}")
    return EMERGENCY_PROTOCOL


# 하위 호환성을 위한 함수 (기존 코드에서 사용 중)
def hybrid_search_milvus(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """하위 호환성을 위한 래퍼 함수"""
    return milvus_knowledge_search.invoke({"query": query, "top_k": top_k})
