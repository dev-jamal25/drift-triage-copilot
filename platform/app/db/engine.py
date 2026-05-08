"""Async engine + session factory wiring.

Lifespan calls ``open_engine`` once and ``close_engine`` on shutdown. The
engine is never recreated per request. Migrations are NOT run from here —
they're an explicit ``uv run alembic upgrade head`` step (see platform/README).
"""

from __future__ import annotations

import structlog
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)

log = structlog.get_logger(__name__)

AsyncSessionFactory = async_sessionmaker  # alias for typing in dependencies.py


def open_engine(database_url: str) -> tuple[AsyncEngine, async_sessionmaker]:
    """Create an async engine + session factory. Call once per process."""
    log.info("db_engine_open", url=_redact(database_url))
    engine = create_async_engine(
        database_url,
        pool_pre_ping=True,
        future=True,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, factory


async def close_engine(engine: AsyncEngine) -> None:
    """Dispose the engine and its connection pool."""
    await engine.dispose()
    log.info("db_engine_closed")


def _redact(url: str) -> str:
    """Drop the password from the URL before logging."""
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host = rest.split("@", 1)
    if ":" in creds:
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"
    return url
