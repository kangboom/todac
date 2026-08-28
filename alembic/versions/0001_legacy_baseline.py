"""Create the legacy TODAC schema for a fresh database.

Existing deployments must stamp this revision because these tables already
exist there. Keeping the DDL explicit makes the historical baseline immutable
when SQLAlchemy models change later.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_legacy_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("nickname", sa.String(50), nullable=False),
        sa.Column("role", sa.Enum("USER", "ADMIN", name="userrole"), nullable=False),
        sa.Column("refresh_token", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_users_id", "users", ["id"])
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "baby_profiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(50), nullable=False),
        sa.Column("birth_date", sa.Date(), nullable=False, comment="실제 태어난 날 (예방접종 기준)"),
        sa.Column("due_date", sa.Date(), nullable=False, comment="출산 예정일 (교정 연령/발달 평가 기준)"),
        sa.Column("gender", sa.String(10), nullable=True, comment="성별: M 또는 F"),
        sa.Column("birth_weight", sa.Float(), nullable=False, comment="출생 체중 (kg)"),
        sa.Column("birth_height", sa.Float(), nullable=True, comment="출생 키 (cm)"),
        sa.Column("medical_history", postgresql.JSONB(), nullable=False, comment="기저질환 리스트 (예: ['RDS', '황달'])"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.CheckConstraint("gender IN ('M', 'F')", name="check_gender"),
    )
    op.create_index("ix_baby_profiles_id", "baby_profiles", ["id"])
    op.create_index("ix_baby_profiles_user_id", "baby_profiles", ["user_id"])

    op.create_table(
        "chat_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("baby_id", postgresql.UUID(as_uuid=True), nullable=False, comment="상담 대상 아기 ID (Context 주입용)"),
        sa.Column("title", sa.String(100), nullable=True, comment="세션 제목 (첫 질문으로 자동 생성)"),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("missing_info", postgresql.JSONB(), nullable=True, comment="부족한 정보 목록 (예: ['아기 월령', '수유량']) - 다음 턴에서 참조"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["baby_id"], ["baby_profiles.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_chat_sessions_id", "chat_sessions", ["id"])
    op.create_index("ix_chat_sessions_user_id", "chat_sessions", ["user_id"])
    op.create_index("ix_chat_sessions_baby_id", "chat_sessions", ["baby_id"])

    op.create_table(
        "chat_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(20), nullable=False, comment="화자: USER 또는 ASSISTANT"),
        sa.Column("content", sa.Text(), nullable=False, comment="대화 내용 텍스트"),
        sa.Column("is_retry", sa.Boolean(), nullable=False, comment="재질문 모드 여부"),
        sa.Column("is_emergency", sa.Boolean(), nullable=False, comment="응급 상황 감지 여부 (통계 분석용)"),
        sa.Column("rag_sources", postgresql.JSONB(), nullable=True, comment="참조 문서 정보 (예: [{'doc_id': '...', 'score': 0.9}])"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"], ondelete="CASCADE"),
        sa.CheckConstraint("role IN ('USER', 'ASSISTANT')", name="check_message_role"),
    )
    op.create_index("ix_chat_messages_id", "chat_messages", ["id"])
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"])
    op.create_index("ix_chat_messages_is_emergency", "chat_messages", ["is_emergency"])
    op.create_index("ix_chat_messages_created_at", "chat_messages", ["created_at"])

    op.create_table(
        "feedbacks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("score", sa.Integer(), nullable=False, comment="만족도 점수 (1~5)"),
        sa.Column("comment", sa.Text(), nullable=True, comment="개선 의견"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["message_id"], ["chat_messages.id"], ondelete="CASCADE"),
        sa.CheckConstraint("score >= 1 AND score <= 5", name="check_feedback_score"),
    )
    op.create_index("ix_feedbacks_id", "feedbacks", ["id"])
    op.create_index("ix_feedbacks_message_id", "feedbacks", ["message_id"])

    op.create_table(
        "knowledge_docs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, comment="문서의 고유 식별자 (자동 생성)"),
        sa.Column("filename", sa.Text(), nullable=False, comment="원본 파일의 이름 (확장자 포함)"),
        sa.Column("storage_url", sa.Text(), nullable=False, unique=True, comment="S3에 저장된 Markdown 파일 경로"),
        sa.Column("raw_pdf_url", sa.Text(), nullable=True, comment="S3에 저장된 원본 PDF 파일의 저장 경로"),
        sa.Column("doc_hash", sa.String(64), nullable=True, comment="문서 내용의 해시값 (중복 업로드 방지)"),
        sa.Column("file_size", sa.Integer(), nullable=True, comment="파일 크기 (bytes)"),
        sa.Column("meta_info", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb"), comment="가변 메타데이터 (작성자, 태그, 카테고리 등)"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, comment="레코드 생성 일시"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, comment="레코드 최종 수정 일시"),
    )
    op.create_index("idx_knowledge_docs_doc_hash", "knowledge_docs", ["doc_hash"])
    op.create_index("ix_knowledge_docs_doc_hash", "knowledge_docs", ["doc_hash"])

    op.create_table(
        "official_qna",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("category", sa.String(), nullable=False, comment="카테고리 (예: 예방접종, 영양, 응급)"),
        sa.Column("question", sa.String(), nullable=False, comment="질문 (Vector Indexing 대상)"),
        sa.Column("answer", sa.Text(), nullable=False, comment="공식 답변"),
        sa.Column("source", sa.String(), nullable=False, comment="출처 (예: 대한신생아학회)"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_official_qna_id", "official_qna", ["id"])
    op.create_index("ix_official_qna_category", "official_qna", ["category"])


def downgrade() -> None:
    op.drop_table("official_qna")
    op.drop_table("knowledge_docs")
    op.drop_table("feedbacks")
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_table("baby_profiles")
    op.drop_table("users")
    op.execute("DROP TYPE IF EXISTS userrole")
