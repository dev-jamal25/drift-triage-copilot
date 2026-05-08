"""End-to-end integration test for the worker against real Redis + Postgres (Step 8).

This test requires Docker services (redis, postgres) to be running.

By default, these tests are skipped to avoid requiring Docker in the standard test run.
To run them locally:

    docker compose up -d redis postgres
    cd worker
    uv run alembic upgrade head
    RUN_DOCKER_TESTS=1 uv run pytest -q -k "test_" tests/test_e2e_replay_test.py

Or from repo root:
    docker compose up -d redis postgres
    docker compose run --rm worker uv run alembic upgrade head
    RUN_DOCKER_TESTS=1 docker compose run --rm worker uv run pytest -q -k "test_" tests/test_e2e_replay_test.py
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import UTC, datetime

import pytest
from redis.asyncio import Redis
from shared.contracts import QueuedAction
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import WorkerSettings
from app.queue.keys import PROCESSING, READY

pytestmark = pytest.mark.requires_docker


@pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1",
    reason="Requires Docker; set RUN_DOCKER_TESTS=1 to run",
)
@pytest.mark.asyncio
async def test_replay_test_round_trip() -> None:
    """Enqueue a replay_test action, run worker briefly, verify job persisted as succeeded.

    Pre-flight:
    - docker compose up -d redis postgres
    - cd worker && uv run alembic upgrade head
    """
    settings = WorkerSettings()
    investigation_id = str(uuid.uuid4())
    action = QueuedAction(
        idempotency_key=f"{investigation_id}:replay_test:v3",
        investigation_id=investigation_id,
        model_name="bank-marketing-classifier",
        action_type="replay_test",
        target_version="v3",
        payload={},
        attempt=0,
        max_attempts=3,
        created_at=datetime.now(UTC),
    )

    # Connect to Redis
    redis_client = Redis.from_url(settings.redis_url, decode_responses=False)
    try:
        # Clean up any prior test residue
        await redis_client.lrem(PROCESSING, 0, action.model_dump_json().encode("utf-8"))

        # Enqueue the action
        payload = action.model_dump_json().encode("utf-8")
        await redis_client.lpush(READY, payload)

        # Run the worker for up to 10 seconds; it processes the message and exits
        from app.main import main as worker_main

        try:
            await asyncio.wait_for(worker_main(), timeout=10.0)
        except TimeoutError:
            # Expected; the worker runs until shutdown, so we timeout
            pass

        # Verify queue state: READY and PROCESSING should be empty
        ready_count = await redis_client.llen(READY)
        processing_count = await redis_client.llen(PROCESSING)
        assert ready_count == 0, f"Expected READY to be empty, but had {ready_count} items"
        assert (
            processing_count == 0
        ), f"Expected PROCESSING to be empty, but had {processing_count} items"

        # Verify Postgres: job should have status='succeeded'
        engine = create_async_engine(settings.database_url, future=True)
        AsyncSession = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with AsyncSession() as session:
                result = await session.execute(
                    text("SELECT status FROM worker_action_jobs WHERE idempotency_key = :key"),
                    {"key": action.idempotency_key},
                )
                row = result.first()
                assert (
                    row is not None
                ), f"Expected job record for key {action.idempotency_key}, but not found"
                assert row[0] == "succeeded", f"Expected status='succeeded', but got {row[0]}"
        finally:
            await engine.dispose()
    finally:
        await redis_client.aclose()


@pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1",
    reason="Requires Docker; set RUN_DOCKER_TESTS=1 to run",
)
@pytest.mark.asyncio
async def test_rollback_blocked_dlq() -> None:
    """Enqueue a rollback action, verify it's blocked-DLQ'd without promotion gate.

    Pre-flight: same as test_replay_test_round_trip
    """
    settings = WorkerSettings()
    investigation_id = str(uuid.uuid4())
    action = QueuedAction(
        idempotency_key=f"{investigation_id}:rollback:v3",
        investigation_id=investigation_id,
        model_name="bank-marketing-classifier",
        action_type="rollback",
        target_version="v3",
        payload={},
        attempt=0,
        max_attempts=3,
        created_at=datetime.now(UTC),
    )

    redis_client = Redis.from_url(settings.redis_url, decode_responses=False)
    try:
        # Clean up
        await redis_client.lrem(PROCESSING, 0, action.model_dump_json().encode("utf-8"))

        # Enqueue
        payload = action.model_dump_json().encode("utf-8")
        await redis_client.lpush(READY, payload)

        # Run worker
        from app.main import main as worker_main

        try:
            await asyncio.wait_for(worker_main(), timeout=10.0)
        except TimeoutError:
            pass

        # Verify Postgres: job should have status='blocked'
        settings = WorkerSettings()
        engine = create_async_engine(settings.database_url, future=True)
        AsyncSession = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with AsyncSession() as session:
                result = await session.execute(
                    text(
                        "SELECT status, last_error FROM worker_action_jobs WHERE idempotency_key = :key"
                    ),
                    {"key": action.idempotency_key},
                )
                row = result.first()
                assert row is not None, f"Expected job record for {action.idempotency_key}"
                assert row[0] == "blocked", f"Expected status='blocked', got {row[0]}"
                # Verify the error message mentions the promotion gate
                assert "rollback" in str(row[1]).lower() or "gate" in str(row[1]).lower()
        finally:
            await engine.dispose()
    finally:
        await redis_client.aclose()
