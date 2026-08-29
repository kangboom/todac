"""Remove the upper bound from coaching goal time horizons."""
from alembic import op

revision = "0003_goal_horizon"
down_revision = "0002_grow_coaching_v2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "check_goal_time_horizon",
        "coaching_goals",
        type_="check",
    )
    op.create_check_constraint(
        "check_goal_time_horizon",
        "coaching_goals",
        "time_horizon_days >= 1",
    )


def downgrade() -> None:
    op.drop_constraint(
        "check_goal_time_horizon",
        "coaching_goals",
        type_="check",
    )
    op.create_check_constraint(
        "check_goal_time_horizon",
        "coaching_goals",
        "time_horizon_days BETWEEN 1 AND 3",
    )
