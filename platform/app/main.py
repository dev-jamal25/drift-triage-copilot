"""Drift Triage Co-Pilot — Model Service entry point."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI, Request

from app.core.logging import configure_logging
from app.db.engine import close_engine, open_engine
from app.dependencies import get_settings
from app.routers import drift as drift_router
from app.routers import predict as predict_router
from app.services.drift_scheduler import DriftScheduler
from app.services.model_loader import load_bundle


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the DB engine, load the model, and start the drift scheduler."""
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

    http_client = httpx.AsyncClient(timeout=settings.agent_webhook_timeout_s)
    app.state.http_client = http_client

    scheduler = DriftScheduler(
        settings=settings,
        session_factory=session_factory,
        get_bundle=lambda: getattr(app.state, "model_bundle", None),
        http_client=http_client,
    )
    scheduler.start()
    app.state.drift_scheduler = scheduler

    try:
        yield
    finally:
        await scheduler.stop()
        await http_client.aclose()
        await close_engine(engine)
        log.info("shutdown")


app = FastAPI(
    title="Drift Triage Co-Pilot — Model Service",
    lifespan=lifespan,
)
app.include_router(predict_router.router)
app.include_router(drift_router.router)


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
