"""Alembic environment for the worker service.

Reads the database URL from app.core.config.WorkerSettings (no
os.getenv here). Uses ``worker_alembic_version`` to avoid colliding
with platform's ``platform_alembic_version`` and MLflow's default
``alembic_version``.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Set up sys.path to find app modules
WORKER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKER_ROOT.parent

for path in (WORKER_ROOT, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from app.core.config import WorkerSettings  # noqa: E402

# Import all models so Base.metadata sees them.
from app.db import models as _models  # noqa: E402, F401
from app.db.base import Base  # noqa: E402

config = context.config

# Skip file-based logging config to avoid issues with alembic.ini.
# We use structlog in the application; migrations don't need complex logging.
# if config.config_file_name is not None:
#     fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _resolve_url() -> str:
    return WorkerSettings().database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_resolve_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table="worker_alembic_version",
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table="worker_alembic_version",
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _resolve_url()
    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
