"""FastAPI dependency providers.

Tests override ``get_model_bundle`` with a fixture-built bundle so they don't
need MLflow. Production wiring stores the bundle on ``app.state`` during
lifespan and returns it via this dependency.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import HTTPException, Request, status

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
