"""Agent service entry point."""

import os
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from agent.app.database import init_db, close_db
from agent.app.queue import get_queue_client
from agent.app.routers import webhooks, approvals, investigations

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: startup and shutdown."""
    log.info("agent.startup")

    # Initialize database
    await init_db()

    # Connect to Redis
    queue_client = get_queue_client()
    await queue_client.connect()

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


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
