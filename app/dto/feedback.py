"""
피드백 관련 DTO
"""
from pydantic import BaseModel, Field, UUID4
from typing import Optional
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
