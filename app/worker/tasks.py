"""
TaskIQ Worker Tasks
문서 처리 등 백그라운드 작업 정의
"""
import uuid
import logging
import json
import boto3
import gc
from app.core.taskiq import broker
from app.core.database import SessionLocal
from app.models.knowledge import KnowledgeDoc
from app.services.parsers.llama_parse_parser import LlamaParseParser
from app.services.parsers.pymupdf_parser import PyMuPDFParser
from app.services.parsers.docling_parser import DoclingParser
from app.services.chunking_markdown import chunk_markdown_documents
from app.services.markdown_service import cleanup_markdown_with_llm
from app.services.chunking_markdown import chunk_markdown_documents
from app.services.markdown_service import cleanup_markdown_with_llm
from app.services.parser_service import get_parser, get_active_parser
from app.services.s3_service import upload_to_s3, delete_from_s3, generate_storage_paths
from app.agent.tools import get_embedding
from app.core.milvus_schema import MILVUS_COLLECTION_NAME
from app.core.database import get_milvus_client
from app.core.config import settings, get_embeddings
from app.core.milvus_schema import create_milvus_collection

logger = logging.getLogger(__name__)

# S3 클라이언트 (boto3)
s3_client = boto3.client(
    's3',
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.S3_REGION
)

@broker.task
async def process_document_task(
    doc_id_str: str,
    raw_s3_key: str,
    filename: str,
    category: str,
    user_id_str: str,
    file_size: int,
    doc_hash: str
):
    """
    백그라운드 문서 처리 태스크
    1. S3에서 원본 파일 다운로드
    2. 문서 파싱 & Markdown 변환
    3. Markdown S3 업로드
    4. 청킹 & 임베딩
    5. Milvus 저장
    6. DB 메타데이터 저장
    """
    logger.info(f"🚀 문서 처리 태스크 시작: doc_id={doc_id_str}, file={filename}")
    
    db = SessionLocal()
    doc_id = uuid.UUID(doc_id_str)
    user_id = uuid.UUID(user_id_str)
    
    # 롤백용 리소스 추적
    uploaded_s3_keys = []
    milvus_inserted = False
    
    try:
        # 1. S3에서 파일 다운로드
        try:
            response = s3_client.get_object(Bucket=settings.S3_BUCKET_NAME, Key=raw_s3_key)
            content = response['Body'].read()
            logger.info(f"S3 다운로드 완료: {len(content)} bytes")
        except Exception as e:
            logger.error(f"S3 파일 다운로드 실패: {e}")
            raise e

        # 2. 파서 찾기
        parser = get_parser(filename)
        if not parser:
            logger.error(f"지원하지 않는 파일 형식: {filename}")
            return # 실패 처리 (DB 업데이트 등 필요할 수 있음)

        # 3. 문서 파싱
        try:
            documents = parser.parse(content, filename)
            # 메모리 절약: 파싱 완료 후 원본 content 삭제
            del content
            gc.collect() 
        except Exception as e:
            logger.error(f"문서 파싱 실패: {e}")
            raise e
            
        # 3-1. Markdown 보정
        if isinstance(parser, (LlamaParseParser, PyMuPDFParser, DoclingParser)):
            if documents and len(documents) > 0:
                original_text = documents[0].text
                if original_text:
                    cleaned_text = cleanup_markdown_with_llm(original_text, filename)
                    documents[0].text = cleaned_text
                    logger.info(f"Markdown 보정 완료: {filename}")

        # 4. 텍스트 청킹
        chunks = chunk_markdown_documents(documents)
        if not chunks:
            raise ValueError("파싱된 텍스트가 없습니다.")

        # 5. Markdown 텍스트 S3 업로드 (Processed)
        markdown_text = documents[0].text if documents else ""
        markdown_bytes = markdown_text.encode('utf-8')
        
        storage_paths = generate_storage_paths(doc_id, filename)
        
        # processed_md_key로 업로드 (raw_s3_key는 이미 API 서버에서 올림)
        storage_url = upload_to_s3(
            content=markdown_bytes,
            s3_key=storage_paths.processed_md_key,
            content_type='text/markdown'
        )
        uploaded_s3_keys.append(storage_url)

        # 6. 임베딩 및 Milvus 저장 (Batch 최적화 적용)
        try:
            logger.info(f"Milvus 컬렉션 생성 시작...")
            create_milvus_collection()
            logger.info(f"Milvus 컬렉션 생성 완료...")

            embeddings = get_embeddings()
            client = get_milvus_client()
            
            embedding_texts = []
            prepared_metadata = [] 

            for chunk in chunks:
                # 헤더 정보 추출
                header_metadata = {
                    k: v for k, v in chunk.metadata.items() 
                    if k.startswith("Header")
                }
                
                # 임베딩용 텍스트 구성
                if header_metadata:
                    sorted_headers = [header_metadata[k] for k in sorted(header_metadata.keys())]
                    header_path = " > ".join(sorted_headers)
                    text_for_embedding = f"{header_path}\n\n{chunk.text}"
                else:
                    text_for_embedding = chunk.text
                
                # 리스트에 추가 (나중에 한방에 변환)
                embedding_texts.append(text_for_embedding)
                
                # 나중에 row 만들 때 매칭할 정보 저장
                prepared_metadata.append({
                    "chunk": chunk,
                    "headers_json": json.dumps(header_metadata, ensure_ascii=False) if header_metadata else "{}"
                })

            # [Step 2] 배치 임베딩 실행 (가장 큰 성능 향상 구간)
            logger.info(f"임베딩 생성 시작 (총 {len(embedding_texts)}개 청크 Batch 처리)...")
            vectors = embeddings.embed_documents(embedding_texts)
            
            # 메모리 절약: 임베딩 생성 후 텍스트 리스트 삭제
            del embedding_texts
            gc.collect()
            
            logger.info("임베딩 생성 완료")

            # [Step 3] 데이터 조립 및 Milvus 배치 저장
            batch_size = 100 
            rows = []
            total_count = 0
            
            for i, vector in enumerate(vectors):
                meta = prepared_metadata[i]
                chunk = meta['chunk']
                
                row = {
                    "doc_id": str(doc_id),
                    "chunk_index": chunk.chunk_index,
                    "embedding": vector,
                    "content": chunk.text[:65535],
                    "filename": filename[:255],
                    "category": category[:50],
                    "headers": meta['headers_json'][:2048]
                }
                rows.append(row)
                total_count += 1
                
                # 배치 단위 저장
                if len(rows) >= batch_size:
                    client.insert(
                        collection_name=MILVUS_COLLECTION_NAME,
                        data=rows
                    )
                    logger.info(f"Milvus 배치 저장: {total_count}/{len(chunks)}개 청크 처리 중...")
                    rows = []

            # 남은 데이터 저장
            if rows:
                client.insert(
                    collection_name=MILVUS_COLLECTION_NAME,
                    data=rows
                )
            
            milvus_inserted = True
            logger.info(f"Milvus 저장 완료: 총 {total_count}개 청크")
            
        except Exception as e:
            logger.error(f"Milvus 저장 실패: {e}")
            raise e

        # 7. DB 저장 (KnowledgeDoc)
        raw_pdf_url = f"s3://{settings.S3_BUCKET_NAME}/{raw_s3_key}"

        meta_info = {
            "category": category,
            "uploaded_by": str(user_id),
            "chunk_count": len(chunks),
            "original_filename": filename,
            "status": "completed"  # 상태 표시
        }
        
        knowledge_doc = KnowledgeDoc(
            id=doc_id,
            filename=filename,
            storage_url=storage_url,
            raw_pdf_url=raw_pdf_url,
            doc_hash=doc_hash,
            file_size=file_size,
            meta_info=meta_info
        )
        
        db.add(knowledge_doc)
        db.commit()
        
        logger.info(f"✅ 문서 처리 완료: doc_id={doc_id}")
        
    except Exception as e:
        logger.error(f"태스크 처리 중 오류 발생: {e}")
        
        # 롤백
        db.rollback()
        
        # S3 삭제
        try:
            # Raw 파일 삭제
            raw_url = f"s3://{settings.S3_BUCKET_NAME}/{raw_s3_key}"
            delete_from_s3(raw_url)
            
            # Processed 파일 삭제
            for url in uploaded_s3_keys:
                delete_from_s3(url)
        except Exception as s3_err:
            logger.error(f"S3 롤백 실패: {s3_err}")
            
        # Milvus 삭제
        if milvus_inserted:
            try:
                client = get_milvus_client()
                client.delete(
                    collection_name=MILVUS_COLLECTION_NAME,
                    filter=f'doc_id == "{doc_id_str}"'
                )
            except Exception as m_err:
                logger.error(f"Milvus 롤백 실패: {m_err}")
        
    finally:
        db.close()
        
        # [메모리 최적화]
        # 1. 딥러닝 모델(Docling) 등 무거운 객체의 캐시를 비움
        try:
            get_active_parser.cache_clear()
            logger.info("Parser 캐시 초기화 완료 (메모리 반환)")
        except Exception:
            pass
            
        # 2. 강제 가비지 컬렉션 수행
        gc.collect()

