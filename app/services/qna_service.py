import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from langchain_openai import OpenAIEmbeddings
from pymilvus import utility

from app.models.qna import OfficialQnA
from app.dto.qna import QnADoc
from app.core.milvus_schema import get_qna_collection_safe, create_qna_collection, OFFICIAL_QNA_COLLECTION_NAME
from app.core.config import settings

logger = logging.getLogger(__name__)

# 임베딩 모델 설정
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=settings.OPENAI_API_KEY
)

def search_qna(query: str, limit: int = 5) -> List[QnADoc]:
    """
    QnA 데이터베이스(Milvus)에서 검색
    """
    try:
        query_embedding = embeddings.embed_query(query)
        collection = get_qna_collection_safe()
        
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["qna_id", "question", "answer", "category", "source"]
        )
        
        qna_docs = []
        for hits in results:
            for hit in hits:
                score = max(0.0, 1.0 - (hit.distance / 2)) # L2 distance to similarity score
                
                doc = QnADoc(
                    id=hit.entity.get("qna_id"),
                    question=hit.entity.get("question"),
                    answer=hit.entity.get("answer"),
                    source=hit.entity.get("source"),
                    category=hit.entity.get("category"),
                    score=score
                )
                qna_docs.append(doc)
                
        return qna_docs
        
    except Exception as e:
        logger.error(f"QnA 검색 실패: {str(e)}", exc_info=True)
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
        
        vector = embeddings.embed_query(question)
        
        collection = get_qna_collection_safe()
        
        data = [
            [db_qna.id],      # qna_id
            [question],       # question
            [answer],         # answer
            [category],       # category
            [source],         # source
            [vector]          # embedding
        ]
        
        collection.insert(data)
        collection.flush()
        
        logger.info(f"QnA 등록 완료: ID={db_qna.id}, 질문={question}")
        
        return db_qna
        
    except Exception as e:
        db.rollback()
        logger.error(f"QnA 등록 실패: {str(e)}")
        raise e

def sync_all_qna_to_milvus(db: Session) -> int:
    """
    DB의 모든 QnA 데이터를 Milvus에 동기화 (기존 컬렉션 삭제 후 재생성)
    """
    try:
        all_qnas = db.query(OfficialQnA).all()
        if not all_qnas:
            logger.info("동기화할 QnA 데이터가 없습니다.")
            return 0
            
        logger.info(f"QnA 동기화 시작: 총 {len(all_qnas)}개 데이터")
        
        get_qna_collection_safe() # 연결 확보
        
        if utility.has_collection(OFFICIAL_QNA_COLLECTION_NAME):
            utility.drop_collection(OFFICIAL_QNA_COLLECTION_NAME)
            logger.info("기존 QnA 컬렉션 삭제 완료")
            
        collection = create_qna_collection()
        
        ids = []
        questions = []
        answers = []
        categories = []
        sources = []
        embeddings_list = []
        
        count = 0
        batch_size = 50
        
        for qna in all_qnas:
            try:
                vector = embeddings.embed_query(qna.question)
                
                ids.append(qna.id)
                questions.append(qna.question)
                answers.append(qna.answer)
                categories.append(qna.category)
                sources.append(qna.source)
                embeddings_list.append(vector)
                
                count += 1
                
                if len(ids) >= batch_size:
                    data = [ids, questions, answers, categories, sources, embeddings_list]
                    collection.insert(data)
                    logger.info(f"{count}/{len(all_qnas)} 처리 중...")
                    
                    ids = []
                    questions = []
                    answers = []
                    categories = []
                    sources = []
                    embeddings_list = []
                    
            except Exception as e:
                logger.error(f"QnA ID {qna.id} 임베딩 실패: {str(e)}")
                continue
        
        if ids:
            data = [ids, questions, answers, categories, sources, embeddings_list]
            collection.insert(data)
        
        collection.flush()
        collection.load()
        
        logger.info(f"QnA 동기화 완료: 총 {count}개 성공")
        return count
        
    except Exception as e:
        logger.error(f"QnA 동기화 전체 실패: {str(e)}")
        raise e
