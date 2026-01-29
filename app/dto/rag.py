from pydantic import BaseModel
from typing import Optional

class RagDoc(BaseModel):
    """RAG 검색 결과 문서 DTO"""
    doc_id: Optional[str] = None
    chunk_index: Optional[int] = None
    content: str
    filename: str
    category: Optional[str] = ""
    score: float = 0.0

