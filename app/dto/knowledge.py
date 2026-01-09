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
    file_size: Optional[int] = None
    meta_info: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """문서 목록 응답"""
    documents: List[DocumentResponse]
    total: int
    limit: int
    offset: int


class BatchDocumentResult(BaseModel):
    """배치 업로드 결과 (단일 문서)"""
    success: bool
    filename: str
    document: Optional[DocumentResponse] = None
    error: Optional[str] = None


class BatchDocumentResponse(BaseModel):
    """배치 문서 업로드 응답"""
    results: List[BatchDocumentResult]
    total: int
    success_count: int
    failure_count: int


class ParsedDocument(BaseModel):
    """파싱된 문서 (청킹 전)"""
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """문서 청크"""
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_index: int


class StoragePath(BaseModel):
    """S3 저장 경로"""
    raw_pdf_key: str
    processed_md_key: str
    images_dir: str


