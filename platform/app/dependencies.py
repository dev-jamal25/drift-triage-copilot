"""FastAPI dependency providers.

Tests override these with fixtures (a fitted ``ModelBundle`` and an
``AsyncMock`` session) so they don't need MLflow or a real DB. Production
wiring stores the bundle and the session factory on ``app.state`` during
lifespan; these providers return them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache

from fastapi import HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.services.model_loader import ModelBundle


@lru_cache
def get_settings() -> Settings:
    """Cached Settings instance — single source of runtime config per process."""
    return Settings()


def get_model_bundle(request: Request) -> ModelBundle:
    """Return the model bundle loaded during lifespan, or 503 if not ready."""
    bundle: ModelBundle | None = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet.",
        )
    return bundle


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield an ``AsyncSession`` bound to the lifespan-opened engine."""
    factory: async_sessionmaker | None = getattr(request.app.state, "session_factory", None)
    if factory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database engine not opened.",
        )
    async with factory() as session:
        yield session
