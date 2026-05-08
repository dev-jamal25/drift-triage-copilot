"""Postgres database session setup and initialization."""

import os

import structlog
from agent.app.models import Base
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

log = structlog.get_logger()


def _get_postgres_dsn() -> str:
    """Construct the Postgres connection string from environment."""
    dsn = os.getenv(
        "POSTGRES_DSN",
        "postgresql+asyncpg://drift_user:drift_pass@postgres:5432/drift_triage",
    )
    return dsn


# Create async engine with connection pooling disabled for Flask-like usage
_engine = None
_session_maker = None


async def init_db() -> None:
    """Initialize the database engine and session maker."""
    global _engine, _session_maker

    dsn = _get_postgres_dsn()
    log.info("database.init", dsn=dsn)

    _engine = create_async_engine(
        dsn,
        echo=False,
        poolclass=NullPool,  # Disable pooling for simplicity; we'll use a single session per request
    )

    _session_maker = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create tables if they don't exist
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """Get a new database session."""
    if _session_maker is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _session_maker()


async def close_db() -> None:
    """Close the database engine."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
