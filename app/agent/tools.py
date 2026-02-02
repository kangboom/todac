"""
Milvus 검색 도구 (Hybrid Search 구현)
"""
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from pymilvus import Collection
from app.services.qna_service import search_qna # [추가]
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


@tool(response_format="content_and_artifact")
def retrieve_qna(query: str) -> str:
    """
    미숙아 및 신생아 관련 공식 QnA(질문-답변) 데이터베이스를 검색합니다.
    검색된 결과는 '질문(Question)'과 '답변(Answer)' 형태로 제공됩니다.
    
    이 도구는 사용자의 질문과 가장 유사한 기존의 전문적인 QnA 세트를 찾을 때 사용합니다.
    
    Args:
        query: 검색할 질문 내용 (예: "미숙아 수유량", "퇴원 후 관리")
        
    Returns:
        JSON 형식의 문자열 리스트 (검색된 QnA 목록)
    """
    try:
        logger.info(f"=== QnA 검색 시작 (Tool) ===")
        logger.info(f"검색 질문: {query}")
        
        # QnA 서비스 호출
        results = search_qna(query)
        
        if not results:
            logger.info("QnA 검색 결과 없음")
            return "검색된 QnA 결과가 없습니다."
            
        # 결과를 JSON 문자열로 변환 (ToolMessage Content용)
        # DTO 객체를 dict로 변환
        serialized_results = []
        logger.info(f"✅[QnA 검색 결과]")
        for doc in results:
            serialized_results.append({
                "id": str(getattr(doc, "id", "")),
                "question": getattr(doc, "question", ""),
                "answer": getattr(doc, "answer", ""),
                "source": getattr(doc, "source", ""),
                "category": getattr(doc, "category", ""),
                "score": getattr(doc, "score", 0.0)
            })
            logger.info(f"{doc.question}")
            
        logger.info(f"QnA 검색 완료: {len(results)}개 항목 반환")
        content = f"QnA 검색 결과: {len(results)}개 항목 반환"
        artifact = serialized_results
        return content, artifact
        
    except Exception as e:
        logger.error(f"QnA 검색 실패: {str(e)}", exc_info=True)
        return "QnA 검색 중 오류가 발생했습니다."


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


@tool(response_format="content_and_artifact")
def milvus_knowledge_search(
    query: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    미숙아 관련 지식베이스에서 전문 의학 정보를 검색합니다.
    
    이 tool은 다음 상황에서 반드시 사용하세요:
    - 미숙아 관련 의학 정보, 증상, 질병, 육아방법에 대한 질문
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
    - "아기가 잠을 잘 못자"
    
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
        
        # 컬렉션 상태 확인
        collection.load()
        
        # 데이터 개수 확인
        num_entities = collection.num_entities
 
        if num_entities == 0:
            logger.warning("⚠️ Milvus 컬렉션에 데이터가 없습니다. 문서를 먼저 업로드해주세요.")
            return []
        
        # 질문 임베딩
        query_embedding = get_embedding(query)
        
        # 검색 파라미터 (데이터가 적을 때 nprobe 조정)
        # nprobe는 nlist보다 작아야 함 (기본 nlist=1024)
        nprobe = min(10, max(1, num_entities // 100 + 1))
        search_params = {
            "metric_type": "L2",  # 유클리드 거리
            "params": {"nprobe": nprobe}
        }
        
        # 벡터 검색 수행 (카테고리 필터 없이 전체 검색)
        logger.info(f"Milvus 벡터 검색 실행 중...")
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",  # 임베딩 필드명
            param=search_params,
            limit=top_k,
            output_fields=["doc_id", "content", "filename", "category", "chunk_index", "headers"]
        )
        
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
                    f"  [{idx+1}] "
                    f"chunk_index={doc['chunk_index']}, "
                    f"filename={doc['filename']}, "
                    f"content_length={len(doc['content'])}"
                )
        
        logger.info(f"=== Milvus 검색 완료: {len(retrieved_docs)}개 문서 검색됨 ===")

        content = f"Milvus 검색 결과: {len(retrieved_docs)}개 문서 검색됨"
        artifact = retrieved_docs
        return content, artifact
        
    except Exception as e:
        logger.error(f"❌ Milvus 검색 실패: {str(e)}", exc_info=True)
        import traceback
        logger.error(f"상세 에러:\n{traceback.format_exc()}")
        # 에러 발생 시 빈 리스트 반환
        return []


