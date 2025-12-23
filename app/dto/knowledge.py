"""
지식 베이스 관련 DTO
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid


class DocumentIngestRequest(BaseModel):
    """문서 업로드 요청"""
    category: str = Field(..., min_length=1, max_length=50, description="문서 카테고리 (예: 식이, 수면, 호흡, 발달, 예방접종, 피부, 응급)")


class DocumentResponse(BaseModel):
    """문서 응답"""
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    filename: str
    storage_url: str
    raw_pdf_url: Optional[str] = None
    doc_hash: Optional[str] = None
    meta_info: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """문서 목록 응답"""
    documents: List[DocumentResponse]
    total: int
    limit: int
    offset: int

