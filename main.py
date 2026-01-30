"""
App 진입점 (FastAPI 인스턴스 생성)
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from app.api.v1 import auth, users, chat, feedback
from app.api.v1.admin import (
    knowledge as admin_knowledge,
    dashboard as admin_dashboard,
    qna as admin_qna  # [추가]
)
from app.core.config import settings
from app.core.database import Base, engine
from app.models import *

# 로깅 설정
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# uvicorn 로거 레벨 조정 (너무 많은 로그 방지)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="TODAC 미숙아 챗봇 API",
    description="미숙아 챗봇 백엔드 API",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI 경로
    redoc_url="/redoc",  # ReDoc 경로
    openapi_url="/openapi.json",  # OpenAPI 스키마 경로
)

# Swagger UI 커스터마이징
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="TODAC 미숙아 챗봇 API",
        version="1.0.0",
        description="""
        ## 미숙아 챗봇 백엔드 API
        
        ### 인증
        - 회원가입: `/api/v1/auth/signup`
        - 로그인: `/api/v1/auth/login`
        - 로그인 후 받은 토큰을 "Authorize" 버튼에 입력하세요.
        
        ### 사용 방법
        1. 회원가입 또는 로그인하여 토큰을 받습니다.
        2. 우측 상단의 "Authorize" 버튼을 클릭합니다.
        3. `Bearer {토큰}` 형식으로 입력합니다 (예: `Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`)
        4. 보호된 엔드포인트를 사용할 수 있습니다.
        """,
        routes=app.routes,
    )
    
    # OAuth2 인증 스키마 추가 (Swagger UI에서 로그인 폼 사용)
    openapi_schema["components"]["securitySchemes"] = {
        "OAuth2PasswordBearer": {
            "type": "oauth2",
            "flows": {
                "password": {
                    "tokenUrl": "/api/v1/auth/token",
                    "scopes": {}
                }
            }
        }
    }
    
    # 보호된 경로에 security 적용 (auth 경로 제외)
    for path, path_item in openapi_schema.get("paths", {}).items():
        # auth 경로는 제외 (signup, login, token 엔드포인트 제외)
        if "/auth/" not in path:
            for method in path_item.keys():
                if method.lower() in ["get", "post", "put", "delete", "patch"]:
                    # security가 없으면 추가
                    if "security" not in path_item[method]:
                        path_item[method]["security"] = [{"OAuth2PasswordBearer": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# CORS 설정
origins = [
    "http://localhost:3000",
    "https://www.todac.cloud",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """헬스 체크 엔드포인트"""
    return {"message": "TODAC API Server", "status": "running"}

@app.get("/health")
async def health_check():
    """상세 헬스 체크"""
    return {"status": "healthy"}

# 라우터 등록
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/v1", tags=["users"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(feedback.router, prefix="/api/v1", tags=["feedback"])
app.include_router(admin_knowledge.router, prefix="/api/v1", tags=["admin"])
app.include_router(admin_dashboard.router, prefix="/api/v1", tags=["admin"])
app.include_router(admin_qna.router, prefix="/api/v1/admin/qna", tags=["admin-qna"]) # [추가]
