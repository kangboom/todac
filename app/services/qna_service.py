import logging
from typing import List
from sqlalchemy.orm import Session
from app.models.qna import OfficialQnA
from app.dto.qna import QnADoc
from app.core.milvus_schema import create_qna_collection, OFFICIAL_QNA_COLLECTION_NAME
from app.core.database import get_milvus_client
from app.core.config import get_embeddings
from pymilvus import AnnSearchRequest, Function, FunctionType

logger = logging.getLogger(__name__)

def search_qna(query: str, limit: int = 5) -> List[QnADoc]:
    """
    [MilvusClient 버전] QnA 하이브리드 검색
    """
    try:
        # 1. 싱글톤 클라이언트 사용
        client = get_milvus_client() 

        # 2. [Dense Search] 요청서 작성 (의미 검색)
        embeddings_client = get_embeddings()
        query_embedding = embeddings_client.embed_query(query)
        
        dense_req = AnnSearchRequest(
            data=[query_embedding],     # 벡터 데이터
            anns_field="embedding",     # 검색할 필드
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=limit * 3
        )

        # 3. [Sparse Search] 요청서 작성 (키워드 검색)
        sparse_req = AnnSearchRequest(
            data=[query],               # 질문 텍스트 그대로 입력
            anns_field="sparse",        # 검색할 필드
            param={"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}},
            limit=limit * 3
        )

        ranker = Function(
            name="rrf",
            input_field_names=[], 
            function_type=FunctionType.RERANK,
            params={
                "reranker": "rrf",
                "k": 100 
            }
        )

        # 5. 검색 실행 (reqs 파라미터 사용)
        results = client.hybrid_search(
            collection_name=OFFICIAL_QNA_COLLECTION_NAME, # 컬렉션 이름 지정
            reqs=[dense_req, sparse_req],                 # 요청서 2개 묶어서 전달
            ranker=ranker,                              # 랭킹 전략 전달
            limit=limit,
            output_fields=["qna_id", "question", "answer", "category", "source"]
        )
        
        # 6. 결과 파싱
        qna_docs = []
        for hits in results:
            for hit in hits:
                
                entity = hit.get('entity', {})
                distance = hit.get('distance', 0.0)
                
                doc = QnADoc(
                    id=entity.get("qna_id"),
                    question=entity.get("question", ""),
                    answer=entity.get("answer", ""),
                    source=entity.get("source", ""),
                    category=entity.get("category", ""),
                    distance=distance
                )
                qna_docs.append(doc)
                
        return qna_docs
        
    except Exception as e:
        logger.error(f"QnA 하이브리드 검색 실패: {str(e)}", exc_info=True)
        return []

def format_qna_docs(docs: List[QnADoc]) -> str:
    """QnA 검색 결과를 프롬프트용 문자열로 변환"""
    formatted = []
    for doc in docs:
        formatted.append(f"Q: {doc.question}\nA: {doc.answer}\n(출처: {doc.source})")
    return "\n\n".join(formatted)

def ingest_qna(db: Session, question: str, answer: str, source: str, category: str) -> OfficialQnA:
    """
    QnA 데이터 등록 (DB 저장 + Milvus 임베딩)
    """
    try:
        db_qna = OfficialQnA(
            question=question,
            answer=answer,
            source=source,
            category=category
        )
        db.add(db_qna)
        db.commit()
        db.refresh(db_qna)
        
        embeddings_client = get_embeddings()
        vector = embeddings_client.embed_query(question)
        
        client = get_milvus_client()
        create_qna_collection()  # 컬렉션이 없으면 생성
        
        # MilvusClient는 딕셔너리 형태로 데이터 삽입
        data = [{
            "qna_id": db_qna.id,
            "question": question,
            "answer": answer,
            "question_answer": f"{question}\n\n{answer}",  # 통합 필드 추가
            "category": category if category else "",
            "source": source if source else "",
            "embedding": vector
        }]
        
        client.insert(
            collection_name=OFFICIAL_QNA_COLLECTION_NAME,
            data=data
        )
        
        logger.info(f"QnA 등록 완료: ID={db_qna.id}, 질문={question}")
        
        return db_qna
        
    except Exception as e:
        db.rollback()
        logger.error(f"QnA 등록 실패: {str(e)}")
        raise e

def sync_all_qna_to_milvus(db: Session) -> int:
    """
    DB의 QnA 데이터를 Milvus에 동기화 (MilvusClient 사용으로 Insert 문제 해결)
    """
    try:
        all_qnas = db.query(OfficialQnA).all()
        if not all_qnas:
            logger.info("동기화할 QnA 데이터가 없습니다.")
            return 0
            
        logger.info(f"QnA 동기화 시작: 총 {len(all_qnas)}개 데이터")

        # 1. 컬렉션 초기화 (기존 코드 유지)
        client = get_milvus_client()
        if client.has_collection(OFFICIAL_QNA_COLLECTION_NAME):
            logger.info("기존 QnA 컬렉션 삭제 중...")
            client.drop_collection(OFFICIAL_QNA_COLLECTION_NAME)
            logger.info("기존 QnA 컬렉션 삭제 완료")
        
        create_qna_collection()
        logger.info(f"QnA 컬렉션 생성 완료...")

        embeddings_client = get_embeddings()

        # [Step 1] 전체 데이터 준비
        embedding_texts = []
        for qna in all_qnas:
            embedding_texts.append(qna.question)
            
        # [Step 2] 배치 임베딩 실행 (Batch 최적화)
        logger.info(f"임베딩 생성 시작 (총 {len(embedding_texts)}개 질문 Batch 처리)...")
        # embed_documents는 내부적으로 chunk_size(200)만큼 나눠서 API를 호출함
        vectors = embeddings_client.embed_documents(embedding_texts)
        logger.info("임베딩 생성 완료")

        # [Step 3] 데이터 조립 및 Milvus 배치 저장
        batch_size = 100
        rows = [] 
        total_count = 0

        for i, qna in enumerate(all_qnas):
            try:
                row = {
                    "qna_id": qna.id,
                    "question": qna.question, 
                    "answer": qna.answer,
                    "question_answer": f"{qna.question}\n\n{qna.answer}",  # 통합 필드 추가
                    "category": qna.category if qna.category else "",
                    "source": qna.source if qna.source else "",
                    "embedding": vectors[i] # 미리 생성한 벡터 사용
                }
                rows.append(row)
                
                total_count += 1
                
                if len(rows) >= batch_size:
                    client.insert(
                        collection_name=OFFICIAL_QNA_COLLECTION_NAME,
                        data=rows
                    )
                    logger.info(f"{total_count}건 처리 중...")
                    rows = [] 

            except Exception as e:
                logger.error(f"ID {qna.id} 처리 중 에러: {e}")
                continue

        # 5. 남은 데이터 처리
        if rows:
            client.insert(
                collection_name=OFFICIAL_QNA_COLLECTION_NAME,
                data=rows
            )
        
        client.load_collection(OFFICIAL_QNA_COLLECTION_NAME)
        
        logger.info(f"QnA 동기화 완료: 총 {total_count}개 문서 저장됨")
            
        return total_count

    except Exception as e:
        logger.error(f"QnA 동기화 전체 실패: {str(e)}")
        raise e