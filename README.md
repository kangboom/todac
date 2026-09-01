# TODAC Backend

**TODAC**의 백엔드 서버입니다. FastAPI를 기반으로 구축되었으며, LangGraph를 이용한 Self-RAG (Retrieval-Augmented Generation) 로직을 처리합니다.

## 🛠️ 기술 스택

*   **Framework**: FastAPI
*   **Language**: Python 3.10+
*   **Database**: PostgreSQL (SQLAlchemy ORM)
*   **Vector DB**: Milvus
*   **AI/LLM**: LangChain, LangGraph, OpenAI (GPT-4o/mini)
*   **Doc Parser**: LlamaParse, PyMuPDF, Docling

## 🚀 실행 방법

### 1. 환경 변수 설정 (.env)
`todac` 폴더 내에 `.env` 파일을 생성하고 아래 내용을 입력하세요.

```ini
# Database
POSTGRES_USER=todac_user
POSTGRES_PASSWORD=todac_password
POSTGRES_DB=todac_db
POSTGRES_PORT=5432

# Security
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# OpenAI & AI Services
OPENAI_API_KEY=sk-...
LLAMAPARSE_API_KEY=llx-...

# Milvus
MILVUS_HOST=milvus
MILVUS_PORT=19530

# GROW 코칭 V2 (첫 배포에서는 false 유지)
COACHING_V2_ENABLED=false
COACHING_MAX_ADJUSTMENTS=3
```

### 2. Docker Compose로 실행 (권장)
PostgreSQL, Milvus, MinIO 등 필요한 인프라와 함께 백엔드 서버를 실행합니다.

```bash
docker-compose up -d --build
```
*   API 서버: `http://localhost:8000`
*   Swagger UI: `http://localhost:8000/docs`
*   Attu (Milvus UI): `http://localhost:3001`

### 데이터베이스 마이그레이션

새 데이터베이스는 `migrate` 서비스가 Alembic 마이그레이션을 자동 실행한 뒤 Backend와 Worker를 시작합니다.
기존 운영 DB처럼 Alembic 도입 전에 생성된 DB는 먼저 백업한 후, 최초 1회 기준선을 표시하고 V2 스키마를 적용합니다.

```bash
docker compose run --rm migrate alembic stamp 0001_legacy_baseline
docker compose run --rm migrate alembic upgrade head
```

첫 배포에서는 `COACHING_V2_ENABLED=false`로 기존 코칭 흐름을 유지하고, 통합 검증 후 `true`로 전환합니다. 회귀가 발생하면 DB를 되돌리지 않고 기능 플래그만 `false`로 바꿔 V1으로 복귀합니다.

### 3. 로컬 개발 환경에서 실행 (선택)
Docker 대신 로컬 Python 환경에서 실행하려면 다음 단계를 따르세요. (DB 및 Milvus는 별도 실행 필요)

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn main:app --reload
```

## 📁 디렉토리 구조

```
todac/
├── app/
│   ├── agent/          # Self-RAG 로직 (LangGraph)
│   ├── api/            # REST API 엔드포인트
│   ├── core/           # 설정, DB 연결, 보안
│   ├── models/         # SQLAlchemy DB 모델
│   ├── services/       # 비즈니스 로직
│   └── main.py         # 앱 진입점
├── docker-compose.yml  # Docker 구성 파일
└── requirements.txt    # Python 의존성 목록
```

## 🔌 주요 API

*   `/api/v1/auth`: 회원가입, 로그인
*   `/api/v1/babies`: 아기 프로필 관리
*   `/api/v1/chat`: 챗봇 대화 및 세션 관리
*   `/api/v1/admin`: 관리자 기능 (대시보드, 지식 관리)




