"""Add GROW coaching V2 business tables and request idempotency."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_grow_coaching_v2"
down_revision = "0001_legacy_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    chat_columns = {column["name"] for column in inspector.get_columns("chat_messages")}
    if "request_id" not in chat_columns:
        op.add_column(
            "chat_messages",
            sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=True, comment="클라이언트 요청 멱등성 ID"),
        )
    chat_indexes = {index["name"] for index in inspector.get_indexes("chat_messages")}
    if "ix_chat_messages_request_id" not in chat_indexes:
        op.create_index("ix_chat_messages_request_id", "chat_messages", ["request_id"])
    chat_constraints = {constraint["name"] for constraint in inspector.get_unique_constraints("chat_messages")}
    if "uq_chat_message_request_role" not in chat_constraints:
        op.create_unique_constraint(
            "uq_chat_message_request_role", "chat_messages", ["session_id", "request_id", "role"]
        )

    # Some development databases were started once with Base.metadata.create_all()
    # before Alembic became the schema owner. In that transitional state the V2
    # tables already exist, while alterations to legacy tables (such as
    # chat_messages.request_id) are still missing. Reuse the existing tables and
    # reconcile the one known legacy type instead of trying to create them again.
    coaching_tables = {
        "coaching_episodes",
        "coaching_goals",
        "coaching_action_attempts",
        "coaching_events",
    }
    if coaching_tables.issubset(set(inspector.get_table_names())):
        goal_columns = {column["name"]: column for column in inspector.get_columns("coaching_goals")}
        confirmed_column = goal_columns.get("confirmed_by_user")
        if confirmed_column is not None and not isinstance(confirmed_column["type"], sa.Boolean):
            goal_checks = {constraint["name"] for constraint in inspector.get_check_constraints("coaching_goals")}
            if "check_goal_confirmed" in goal_checks:
                op.drop_constraint("check_goal_confirmed", "coaching_goals", type_="check")
            op.alter_column(
                "coaching_goals",
                "confirmed_by_user",
                existing_type=confirmed_column["type"],
                type_=sa.Boolean(),
                existing_nullable=False,
                postgresql_using="(confirmed_by_user <> 0)",
            )
        return

    op.create_table(
        "coaching_episodes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("chat_session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("phase", sa.String(20), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("active_goal_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("pending_interaction", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["chat_session_id"], ["chat_sessions.id"], ondelete="CASCADE"),
        sa.CheckConstraint("attempt_count >= 0", name="check_episode_attempt_count"),
        sa.CheckConstraint("version >= 1", name="check_episode_version"),
    )
    op.create_index("ix_coaching_episodes_chat_session_id", "coaching_episodes", ["chat_session_id"])
    op.create_index(
        "uq_active_episode_per_chat_session",
        "coaching_episodes",
        ["chat_session_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('PENDING_CONSENT', 'ACTIVE', 'WAITING_USER')"),
    )

    op.create_table(
        "coaching_goals",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("episode_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("success_criteria", sa.Text(), nullable=False),
        sa.Column("time_horizon_days", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("confirmed_by_user", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["episode_id"], ["coaching_episodes.id"], ondelete="CASCADE"),
        sa.CheckConstraint("time_horizon_days BETWEEN 1 AND 3", name="check_goal_time_horizon"),
    )
    op.create_index("ix_coaching_goals_episode_id", "coaching_goals", ["episode_id"])
    op.create_foreign_key(
        "fk_episode_active_goal", "coaching_episodes", "coaching_goals", ["active_goal_id"], ["id"], ondelete="SET NULL"
    )

    op.create_table(
        "coaching_action_attempts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("goal_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("selected_action", sa.Text(), nullable=False),
        sa.Column("action_plan", postgresql.JSONB(), nullable=False),
        sa.Column("confidence_score", sa.Integer(), nullable=False),
        sa.Column("result", sa.Text(), nullable=True),
        sa.Column("barrier", sa.Text(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("reported_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["goal_id"], ["coaching_goals.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("goal_id", "sequence", name="uq_action_attempt_sequence"),
        sa.CheckConstraint("sequence >= 1", name="check_action_attempt_sequence"),
        sa.CheckConstraint("confidence_score BETWEEN 0 AND 10", name="check_action_confidence"),
    )
    op.create_index("ix_coaching_action_attempts_goal_id", "coaching_action_attempts", ["goal_id"])

    op.create_table(
        "coaching_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("episode_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("phase", sa.String(20), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["episode_id"], ["coaching_episodes.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("episode_id", "request_id", "event_type", name="uq_coaching_event_request_type"),
    )
    op.create_index("ix_coaching_events_episode_id", "coaching_events", ["episode_id"])
    op.create_index("ix_coaching_events_request_id", "coaching_events", ["request_id"])


def downgrade() -> None:
    op.drop_table("coaching_events")
    op.drop_table("coaching_action_attempts")
    op.drop_constraint("fk_episode_active_goal", "coaching_episodes", type_="foreignkey")
    op.drop_table("coaching_goals")
    op.drop_table("coaching_episodes")
    op.drop_constraint("uq_chat_message_request_role", "chat_messages", type_="unique")
    op.drop_index("ix_chat_messages_request_id", table_name="chat_messages")
    op.drop_column("chat_messages", "request_id")
