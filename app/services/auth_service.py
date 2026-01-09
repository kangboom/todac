"""
회원가입 처리, 비밀번호 해싱
"""
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.models.user import User, UserRole
from app.core.security import get_password_hash, verify_password, create_access_token, create_refresh_token, decode_access_token
from app.dto.auth import SignupRequest, LoginRequest
from datetime import timedelta
from app.core.config import settings
import uuid


class AuthService:
    """인증 서비스"""

    @staticmethod
    def signup(db: Session, request: SignupRequest) -> User:
        """회원가입"""
        # 이메일 중복 확인
        existing_user = db.query(User).filter(User.email == request.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="이미 등록된 이메일입니다."
            )
        
        # 새 사용자 생성
        new_user = User(
            email=request.email,
            password_hash=get_password_hash(request.password),
            nickname=request.nickname,
            role=UserRole.USER
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # 회원가입 후 자동 로그인을 위해 Refresh Token 발급 및 저장
        token_data = {"sub": str(new_user.id)}
        refresh_token = create_refresh_token(token_data)
        
        new_user.refresh_token = refresh_token
        db.commit()
        db.refresh(new_user)
        
        return new_user

    @staticmethod
    def login(db: Session, request: LoginRequest) -> tuple[User, str, str]:
        """로그인"""
        # 사용자 조회
        user = db.query(User).filter(User.email == request.email).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="이메일 또는 비밀번호가 올바르지 않습니다."
            )
        
        # 비밀번호 검증
        if not verify_password(request.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="이메일 또는 비밀번호가 올바르지 않습니다."
            )
        
        # 토큰 생성
        token_data = {
            "sub": str(user.id),
            "email": user.email,
            "role": user.role.value
        }
        
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token({"sub": str(user.id)})
        
        # Refresh Token DB 업데이트
        user.refresh_token = refresh_token
        db.commit()
        
        return user, access_token, refresh_token

    @staticmethod
    def refresh_access_token(db: Session, refresh_token: str) -> str:
        """토큰 갱신"""
        # 1. 토큰 자체 유효성 검증
        payload = decode_access_token(refresh_token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="유효하지 않은 Refresh Token입니다."
            )
            
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="토큰에 사용자 정보가 없습니다."
            )
            
        # 2. DB에 저장된 토큰과 일치하는지 확인 (토큰 탈취 방지 및 강제 로그아웃 지원)
        user = db.query(User).filter(User.id == uuid.UUID(user_id)).first()
        if not user or user.refresh_token != refresh_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh Token이 일치하지 않거나 만료되었습니다. 다시 로그인해주세요."
            )
            
        # 3. 새로운 Access Token 발급
        token_data = {
            "sub": str(user.id),
            "email": user.email,
            "role": user.role.value
        }
        return create_access_token(token_data)


auth_service = AuthService()