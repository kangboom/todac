"""Stop requiring confidence scores for coaching action attempts."""
from alembic import op
import sqlalchemy as sa

revision = "0004_optional_confidence"
down_revision = "0003_goal_horizon"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {
        column["name"]: column
        for column in inspector.get_columns("coaching_action_attempts")
    }
    confidence_column = columns.get("confidence_score")
    if confidence_column is None:
        op.add_column(
            "coaching_action_attempts",
            sa.Column("confidence_score", sa.Integer(), nullable=True),
        )
    elif not confidence_column["nullable"]:
        op.alter_column(
            "coaching_action_attempts",
            "confidence_score",
            existing_type=sa.Integer(),
            nullable=True,
        )


def downgrade() -> None:
    op.execute(
        sa.text(
            "UPDATE coaching_action_attempts "
            "SET confidence_score = 0 WHERE confidence_score IS NULL"
        )
    )
    op.alter_column(
        "coaching_action_attempts",
        "confidence_score",
        existing_type=sa.Integer(),
        nullable=False,
    )
