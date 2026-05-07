"""Drift Triage Co-Pilot — Model Service entry point."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request

from app.core.logging import configure_logging
from app.db.engine import close_engine, open_engine
from app.dependencies import get_settings
from app.routers import predict as predict_router
from app.services.model_loader import load_bundle


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the DB engine and load the registered model once at startup."""
    settings = get_settings()
    configure_logging(settings.log_level)
    log = structlog.get_logger("app.startup")
    log.info(
        "startup",
        mlflow_uri=settings.mlflow_tracking_uri,
        model_name=settings.model_name,
        model_alias=settings.model_alias,
    )

    engine, session_factory = open_engine(settings.database_url)
    app.state.engine = engine
    app.state.session_factory = session_factory

    if settings.load_model_on_startup:
        bundle = await asyncio.to_thread(load_bundle, settings)
        app.state.model_bundle = bundle
    else:
        log.info("model_load_skipped", reason="load_model_on_startup=false")

    try:
        yield
    finally:
        await close_engine(engine)
        log.info("shutdown")


app = FastAPI(
    title="Drift Triage Co-Pilot — Model Service",
    lifespan=lifespan,
)
app.include_router(predict_router.router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness — process is up."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request) -> dict[str, str]:
    """Readiness — model is loaded and the app can serve /predict."""
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        return {"status": "not_ready", "model": "not_loaded"}
    return {
        "status": "ready",
        "model": bundle.model_uri,
        "model_version": bundle.version,
    }
