"""
회원가입 처리, 비밀번호 해싱
"""
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.models.user import User, UserRole
from app.core.security import get_password_hash, verify_password, create_access_token
from app.dto.auth import SignupRequest, LoginRequest
from datetime import timedelta
from app.core.config import settings


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
        
        return new_user

    @staticmethod
    def login(db: Session, request: LoginRequest) -> tuple[User, str]:
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
        
        # JWT 토큰 생성
        token_data = {
            "sub": str(user.id),
            "email": user.email,
            "role": user.role.value
        }
        expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(token_data, expires_delta)
        
        return user, access_token


auth_service = AuthService()