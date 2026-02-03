from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class QnADoc(BaseModel):
    """QnA 검색 결과 문서 DTO (Service -> Agent)"""
    id: Optional[int] = None
    question: str
    answer: str
    source: str
    category: str
    distance: float

class QnACreateRequest(BaseModel):
    """QnA 생성 요청 DTO (API -> Service)"""
    question: str
    answer: str
    source: str
    category: str

class QnAResponse(BaseModel):
    """QnA 응답 DTO (API Response)"""
    id: int
    question: str
    answer: str
    source: str
    category: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True # Pydantic v2 (ORM Mode)
