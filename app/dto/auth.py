"""
로그인/회원가입 요청 양식
"""
from pydantic import BaseModel, EmailStr, Field, field_validator, ConfigDict
from typing import Optional
from datetime import datetime
import uuid


class SignupRequest(BaseModel):
    """회원가입 요청"""
    email: EmailStr
    password: str = Field(..., min_length=8, description="비밀번호 (최소 8자)")
    nickname: str = Field(..., min_length=1, max_length=50, description="사용자 호칭")

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError('비밀번호는 최소 8자 이상이어야 합니다.')
        return v


class LoginRequest(BaseModel):
    """로그인 요청"""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """토큰 응답"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # 초 단위


class UserResponse(BaseModel):
    """사용자 정보 응답"""
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    email: str
    nickname: str
    role: str
    created_at: datetime


class AuthResponse(BaseModel):
    """인증 응답 (토큰 + 사용자 정보)"""
    token: TokenResponse
    user: UserResponse