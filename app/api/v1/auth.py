"""
POST /signup, /login
"""
from fastapi import APIRouter, Depends, HTTPException, status, Body, Response, Cookie
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.dto.auth import SignupRequest, LoginRequest, AuthResponse, TokenResponse, UserResponse
from app.services.auth_service import auth_service
from app.core.config import settings
from datetime import timedelta
from typing import Optional

router = APIRouter()


def set_refresh_token_cookie(response: Response, refresh_token: str):
    """HttpOnly 쿠키에 Refresh Token 설정"""
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=False, # 개발 환경에서는 False (HTTPS 적용 시 True)
        samesite="lax",
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
    )


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def signup(request: SignupRequest, response: Response, db: Session = Depends(get_db)):
    """
    회원가입
    
    - **email**: 이메일 주소 (로그인 아이디)
    - **password**: 비밀번호 (최소 8자)
    - **nickname**: 사용자 호칭 (예: 튼튼맘)
    
    회원가입 성공 시 자동으로 로그인되어 토큰을 받습니다.
    Refresh Token은 HttpOnly 쿠키로 설정됩니다.
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
    
    # HttpOnly 쿠키 설정
    if user.refresh_token:
        set_refresh_token_cookie(response, user.refresh_token)
    
    return AuthResponse(
        token=TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        ),
        user=UserResponse.model_validate(user)
    )


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest, response: Response, db: Session = Depends(get_db)):
    """
    로그인 (JSON 형식)
    
    - **email**: 이메일 주소
    - **password**: 비밀번호
    
    로그인 성공 시 JWT Access Token을 받습니다.
    Refresh Token은 HttpOnly 쿠키로 설정됩니다.
    """
    user, access_token, refresh_token = auth_service.login(db, request)
    
    # HttpOnly 쿠키 설정
    set_refresh_token_cookie(response, refresh_token)
    
    return AuthResponse(
        token=TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        ),
        user=UserResponse.model_validate(user)
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    response: Response,
    refresh_token: Optional[str] = Cookie(None),
    db: Session = Depends(get_db)
):
    """
    Access Token 갱신
    
    - **Cookie**: HttpOnly 쿠키에 담긴 `refresh_token`
    
    유효한 Refresh Token이면 새로운 Access Token을 반환합니다.
    """
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh Token이 없습니다."
        )

    new_access_token = auth_service.refresh_access_token(db, refresh_token)
    
    # Refresh Token 유지 (만료 기간 갱신 등을 원하면 여기서 다시 set_cookie 호출)
    # Refresh Token Rotation (보안 강화): 매 갱신마다 새로운 Refresh Token 발급
    # 새로운 Refresh Token 생성 (유효기간 초기화)
    # 하지만 여기서는 기존 auth_service.refresh_access_token이 access_token만 반환하므로 로직 수정 필요
    # auth_service.refresh_token 메서드를 수정하여 new_refresh_token도 반환하게 해야 함.
    # 일단 현재 구조에서는 Access Token만 갱신하고 Refresh Token 만료 시간만 연장하는 방식을 사용하겠습니다.
    
    # 쿠키 만료 시간만 연장 (Same Refresh Token, Extended Expiry)
    set_refresh_token_cookie(response, refresh_token)
    
    return TokenResponse(
        access_token=new_access_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
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