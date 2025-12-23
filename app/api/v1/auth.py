"""
POST /signup, /login
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.dto.auth import SignupRequest, LoginRequest, AuthResponse, TokenResponse, UserResponse
from app.services.auth_service import auth_service
from app.core.config import settings
from datetime import timedelta

router = APIRouter()


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def signup(request: SignupRequest, db: Session = Depends(get_db)):
    """
    회원가입
    
    - **email**: 이메일 주소 (로그인 아이디)
    - **password**: 비밀번호 (최소 8자)
    - **nickname**: 사용자 호칭 (예: 튼튼맘)
    
    회원가입 성공 시 자동으로 로그인되어 토큰을 받습니다.
    """
    # 회원가입 처리
    user = auth_service.signup(db, request)
    
    # 로그인 토큰 생성
    token_data = {
        "sub": str(user.id),
        "email": user.email,
        "role": user.role.value
    }
    expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    from app.core.security import create_access_token
    access_token = create_access_token(token_data, expires_delta)
    
    return AuthResponse(
        token=TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        ),
        user=UserResponse.model_validate(user)
    )


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """
    로그인 (JSON 형식)
    
    - **email**: 이메일 주소
    - **password**: 비밀번호
    
    로그인 성공 시 JWT 토큰을 받습니다.
    """
    user, access_token = auth_service.login(db, request)
    
    return AuthResponse(
        token=TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        ),
        user=UserResponse.model_validate(user)
    )


@router.post("/token", response_model=dict)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    OAuth2 형식 로그인 (Swagger UI용)
    
    - **username**: 이메일 주소 (OAuth2 표준에서는 username 필드 사용)
    - **password**: 비밀번호
    
    Swagger UI의 "Authorize" 버튼에서 사용됩니다.
    """
    # OAuth2PasswordRequestForm은 username과 password를 받음
    # 우리는 email을 username으로 사용
    login_request = LoginRequest(email=form_data.username, password=form_data.password)
    user, access_token = auth_service.login(db, login_request)
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }