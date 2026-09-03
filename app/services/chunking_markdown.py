"""
Markdown 문서 청킹 유틸리티 (Langchain MarkdownHeaderTextSplitter 사용)
"""
from typing import List, Dict, Any
from langchain_text_splitters import MarkdownHeaderTextSplitter
import logging
from app.dto.knowledge import ParsedDocument, Chunk

logger = logging.getLogger(__name__)


def chunk_markdown_documents(
    documents: List[ParsedDocument],
    chunk_size: int = 600,
    chunk_overlap: int = 120
) -> List[Chunk]:
    """
    Markdown 문서를 헤더 기반으로 청킹
    
    Langchain의 MarkdownHeaderTextSplitter를 사용하여
    Markdown 헤더(#, ##, ### 등)를 기준으로 문서를 분할합니다.
    
    Args:
        documents: ParsedDocument 리스트
        chunk_size: 각 청크의 최대 크기 (헤더 분할 후 추가 분할 시 사용)
        chunk_overlap: 청크 간 겹치는 문자 수
    
    Returns:
        Chunk 리스트
    """
    # MarkdownHeaderTextSplitter 초기화
    # 헤더 레벨별로 분할 (h1, h2, h3)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False  # 헤더를 청크에 포함
    )
    
    all_chunks = []
    
    for doc in documents:
        markdown_text = doc.text
        base_metadata = doc.metadata
        
        if not markdown_text.strip():
            continue
        
        try:
            # 헤더 기반으로 분할
            md_header_splits = splitter.split_text(markdown_text)
            
            # 각 헤더 섹션을 청크로 변환
            for idx, md_chunk in enumerate(md_header_splits):
                chunk_text = md_chunk.page_content
                chunk_metadata = md_chunk.metadata.copy()
                
                # 기본 메타데이터 병합
                chunk_metadata.update(base_metadata)
                
                # 청크가 너무 크면 추가로 분할
                if len(chunk_text) > chunk_size:
                    # 간단한 텍스트 분할 (문장 경계 고려)
                    sub_chunks = _split_large_chunk(chunk_text, chunk_size, chunk_overlap)
                    
                    for sub_idx, sub_chunk in enumerate(sub_chunks):
                        all_chunks.append(Chunk(
                            text=sub_chunk,
                            metadata=chunk_metadata.copy(),
                            chunk_index=idx * 1000 + sub_idx  # 헤더 인덱스 + 서브 인덱스
                        ))
                else:
                    all_chunks.append(Chunk(
                        text=chunk_text,
                        metadata=chunk_metadata,
                        chunk_index=idx
                    ))
            
            logger.info(f"Markdown 청킹 완료: {len(md_header_splits)}개 헤더 섹션 → {len(all_chunks)}개 청크")
            
        except Exception as e:
            logger.error(f"Markdown 청킹 실패: {str(e)}")
            # 실패 시 전체 텍스트를 하나의 청크로 처리
            all_chunks.append(Chunk(
                text=markdown_text,
                metadata=base_metadata.copy(),
                chunk_index=0
            ))
    
    logger.info(f"전체 문서 청킹 완료: {len(documents)}개 문서 → {len(all_chunks)}개 청크")
    
    return all_chunks


def _split_large_chunk(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """
    큰 청크를 작은 청크로 분할 (문장 경계 고려)
    
    Args:
        text: 분할할 텍스트
        chunk_size: 각 청크의 최대 크기
        chunk_overlap: 청크 간 겹치는 문자 수
    
    Returns:
        청크 리스트
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")

    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # 문장 경계에서 자르기
        if end < len(text):
            # 마지막 문장 끝 찾기
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            
            # 겹침을 빼도 다음 시작이 전진하는 경계만 사용합니다.
            last_boundary = max(last_period, last_newline)
            if last_boundary > start and last_boundary + 1 - start > chunk_overlap:
                end = last_boundary + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # 마지막 구간을 처리했으면 겹침 부분을 다시 추가하지 않습니다.
        if end == len(text):
            break

        # 위 경계 조건과 chunk_overlap < chunk_size로 전진이 보장됩니다.
        start = end - chunk_overlap
    
    return chunks

