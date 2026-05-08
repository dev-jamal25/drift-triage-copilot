"""Agent service entry point."""

from contextlib import asynccontextmanager

import structlog
from agent.app.core.config import AgentSettings, redact_url
from agent.app.database import close_db, init_db
from agent.app.queue.client import QueueClient
from agent.app.routers import approvals, investigations, webhooks, worker_jobs
from fastapi import FastAPI

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: startup and shutdown."""
    log.info("agent.startup")

    # Load settings from environment and .env
    settings = AgentSettings()
    log.info(
        "agent.settings.loaded",
        database_url=redact_url(settings.database_url),
        redis_url=redact_url(settings.redis_url),
        log_level=settings.log_level,
    )

    # Initialize database
    await init_db(database_url=settings.database_url)

    # Connect to Redis
    queue_client = QueueClient(redis_url=settings.redis_url)
    await queue_client.connect()

    # Store in app state for dependency injection if needed
    app.state.settings = settings
    app.state.queue_client = queue_client

    yield

    log.info("agent.shutdown")

    # Close connections
    await close_db()
    await queue_client.disconnect()


app = FastAPI(
    title="Drift Triage Co-Pilot — Agent",
    description="LangGraph supervisor for drift investigation and response.",
    lifespan=lifespan,
)

# Register routers
app.include_router(webhooks.router)
app.include_router(approvals.router)
app.include_router(investigations.router)
app.include_router(worker_jobs.router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
