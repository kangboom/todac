"""
텍스트 청킹 유틸리티
"""
from typing import List, Dict, Any
import logging
from app.dto.knowledge import ParsedDocument, Chunk

logger = logging.getLogger(__name__)


# 기본 청킹 설정
DEFAULT_CHUNK_SIZE = 600  # 문자 수
DEFAULT_CHUNK_OVERLAP = 120  # 겹치는 문자 수


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[str]:
    """
    텍스트를 작은 청크로 분할
    
    Args:
        text: 분할할 텍스트
        chunk_size: 각 청크의 최대 크기
        chunk_overlap: 청크 간 겹치는 문자 수
    
    Returns:
        청크 리스트
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 문장 경계에서 자르기 (개선)
        if end < len(text):
            # 마지막 문장 끝 찾기
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            
            # 문장 경계가 있으면 그곳에서 자르기
            if last_period > start and last_period > last_newline:
                end = last_period + 1
            elif last_newline > start:
                end = last_newline + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # 다음 청크 시작 위치 (겹침 고려)
        start = end - chunk_overlap
        if start >= len(text):
            break
    
    return chunks


def chunk_documents(
    documents: List[ParsedDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Chunk]:
    """
    문서 리스트를 청크로 분할
    
    Args:
        documents: ParsedDocument 리스트
        chunk_size: 각 청크의 최대 크기
        chunk_overlap: 청크 간 겹치는 문자 수
    
    Returns:
        Chunk 리스트
    """
    all_chunks = []
    
    for doc in documents:
        text = doc.text
        metadata = doc.metadata
        
        # 텍스트를 청크로 분할
        text_chunks = chunk_text(text, chunk_size, chunk_overlap)
        
        # 각 청크에 메타데이터 추가
        for idx, chunk_text in enumerate(text_chunks):
            all_chunks.append(Chunk(
                text=chunk_text,
                metadata=metadata.copy(),
                chunk_index=idx
            ))
    
    logger.info(f"문서 청킹 완료: {len(documents)}개 문서 → {len(all_chunks)}개 청크")
    
    return all_chunks

