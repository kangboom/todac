from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.core.config import settings
from app.core.database import Base
from app import models  # noqa: F401 - 모델 메타데이터 등록

config = context.config
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL.replace("%", "%%"))

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
LANGGRAPH_TABLES = {
    "checkpoint_migrations",
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
}


def include_object(obj, name, type_, reflected, compare_to):
    """LangGraph가 자체 버전으로 관리하는 체크포인트 테이블은 Alembic 대상에서 제외한다."""
    table_name = name if type_ == "table" else getattr(getattr(obj, "table", None), "name", None)
    return table_name not in LANGGRAPH_TABLES


def run_migrations_offline() -> None:
    context.configure(
        url=settings.DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
