"""Database engine lifecycle management."""

from __future__ import annotations

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import redact_url

log = structlog.get_logger(__name__)


async def open_engine(database_url: str) -> tuple[AsyncEngine, async_sessionmaker]:
    """Create an async SQLAlchemy engine and session factory.

    Returns (engine, session_factory).
    The engine's pool is pre-pinged; future=True for SQLAlchemy 2.0 style.
    """
    engine = create_async_engine(
        database_url,
        future=True,
        pool_pre_ping=True,
        poolclass=NullPool,  # use for long-lived, multi-process scenarios
    )
    # Test connection
    async with engine.begin() as conn:
        await conn.execute(text("SELECT 1"))
    session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )
    log.info("db_engine_open", url=redact_url(database_url))
    return engine, session_factory


async def close_engine(engine: AsyncEngine) -> None:
    """Close the engine and dispose of the pool."""
    await engine.dispose()
    log.info("db_engine_closed")
