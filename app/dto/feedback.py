"""
피드백 관련 DTO
"""
from pydantic import BaseModel, Field, UUID4
from typing import Optional, List
from datetime import datetime


class FeedbackCreateRequest(BaseModel):
    """피드백 생성 요청"""
    message_id: UUID4 = Field(..., description="대상 메시지 ID")
    score: int = Field(..., ge=1, le=5, description="만족도 점수 (1~5)")
    comment: Optional[str] = Field(None, description="상세 의견")


class FeedbackResponse(BaseModel):
    """피드백 응답"""
    id: UUID4
    message_id: UUID4
    score: int
    comment: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class AdminFeedbackResponse(BaseModel):
    """관리자용 피드백 응답 (질문/답변 포함)"""
    id: UUID4
    score: int
    comment: Optional[str]
    created_at: datetime
    
    # 메시지 정보
    message_id: UUID4
    answer: str
    question: Optional[str] = None
    
    # 세션 정보
    session_id: UUID4
    
    # 사용자 정보 [추가]
    user_email: Optional[str] = None
    user_nickname: Optional[str] = None
    
    class Config:
        from_attributes = True

class FeedbackListResponse(BaseModel):
    """피드백 목록 응답"""
    items: List[AdminFeedbackResponse]
    total: int
    page: int
    size: int
