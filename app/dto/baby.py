"""
아기 프로필 요청/응답 양식
"""
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import date, datetime
from typing import List, Optional
import uuid


class BabyCreateRequest(BaseModel):
    """아기 프로필 생성 요청"""
    name: str = Field(..., min_length=1, max_length=50, description="아기 이름/태명")
    birth_date: date = Field(..., description="실제 태어난 날 (예방접종 기준)")
    due_date: date = Field(..., description="출산 예정일 (교정 연령/발달 평가 기준)")
    gender: Optional[str] = Field(None, description="성별: M 또는 F")
    birth_weight: float = Field(..., gt=0, description="출생 체중 (kg)")
    medical_history: List[str] = Field(default_factory=list, description="기저질환 리스트 (예: ['RDS', '황달'])")

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ['M', 'F']:
            raise ValueError('성별은 M 또는 F만 가능합니다.')
        return v

    @field_validator('birth_weight')
    @classmethod
    def validate_birth_weight(cls, v: float) -> float:
        if v <= 0:
            raise ValueError('출생 체중은 0보다 커야 합니다.')
        if v > 10:  # 현실적인 최대값
            raise ValueError('출생 체중이 비정상적으로 큽니다.')
        return v


class BabyUpdateRequest(BaseModel):
    """아기 프로필 수정 요청"""
    name: Optional[str] = Field(None, min_length=1, max_length=50)
    birth_date: Optional[date] = None
    due_date: Optional[date] = None
    gender: Optional[str] = None
    birth_weight: Optional[float] = Field(None, gt=0)
    medical_history: Optional[List[str]] = None

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ['M', 'F']:
            raise ValueError('성별은 M 또는 F만 가능합니다.')
        return v


class BabyResponse(BaseModel):
    """아기 프로필 응답"""
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    user_id: uuid.UUID
    name: str
    birth_date: date
    due_date: date
    gender: Optional[str]
    birth_weight: float
    medical_history: List[str]
    created_at: datetime
    updated_at: datetime

