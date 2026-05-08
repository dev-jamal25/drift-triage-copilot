"""Unit tests for IdempotencyGuard (Step 7)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from shared.contracts import QueuedAction

from app.db.repository import JobsRepository
from app.runtime.idempotency import GuardOutcome, IdempotencyGuard


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    return AsyncMock()


@pytest.fixture
def mock_jobs():
    """Mock JobsRepository."""
    return AsyncMock(spec=JobsRepository)


@pytest.fixture
def guard(mock_redis, mock_jobs):
    """IdempotencyGuard with mocked dependencies."""
    return IdempotencyGuard(mock_redis, mock_jobs)


@pytest.fixture
def sample_action():
    """Sample QueuedAction for testing."""
    return QueuedAction(
        idempotency_key="test-inv-123:replay_test:v3",
        investigation_id="test-inv-123",
        model_name="bank-marketing-classifier",
        action_type="replay_test",
        target_version="v3",
        payload={},
        attempt=0,
        max_attempts=3,
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_acquire_returns_cached_success_when_postgres_has_succeeded(
    guard: IdempotencyGuard, mock_jobs: AsyncMock, sample_action: QueuedAction
) -> None:
    """Test that acquire returns CACHED_SUCCESS when Postgres status='succeeded'."""
    mock_jobs.get_status.return_value = "succeeded"

    outcome = await guard.acquire(sample_action)

    assert outcome == GuardOutcome.CACHED_SUCCESS
    mock_jobs.get_status.assert_called_once_with(sample_action.idempotency_key)


@pytest.mark.asyncio
async def test_acquire_returns_proceed_when_no_existing_row(
    guard: IdempotencyGuard,
    mock_jobs: AsyncMock,
    mock_redis: AsyncMock,
    sample_action: QueuedAction,
) -> None:
    """Test that acquire returns PROCEED when no row exists and SETNX succeeds."""
    mock_jobs.get_status.return_value = None
    mock_redis.set.return_value = True

    outcome = await guard.acquire(sample_action)

    assert outcome == GuardOutcome.PROCEED
    mock_redis.set.assert_called_once()


@pytest.mark.asyncio
async def test_acquire_returns_proceed_when_setnx_succeeds(
    guard: IdempotencyGuard,
    mock_jobs: AsyncMock,
    mock_redis: AsyncMock,
    sample_action: QueuedAction,
) -> None:
    """Test that acquire returns PROCEED when SETNX lock acquired."""
    mock_jobs.get_status.return_value = "pending"
    mock_redis.set.return_value = True

    outcome = await guard.acquire(sample_action)

    assert outcome == GuardOutcome.PROCEED


@pytest.mark.asyncio
async def test_acquire_returns_lock_busy_when_setnx_fails(
    guard: IdempotencyGuard,
    mock_jobs: AsyncMock,
    mock_redis: AsyncMock,
    sample_action: QueuedAction,
) -> None:
    """Test that acquire returns LOCK_BUSY when SETNX fails."""
    mock_jobs.get_status.return_value = "pending"
    mock_redis.set.return_value = False

    outcome = await guard.acquire(sample_action)

    assert outcome == GuardOutcome.LOCK_BUSY


@pytest.mark.asyncio
async def test_release_deletes_lock_key(
    guard: IdempotencyGuard, mock_redis: AsyncMock, sample_action: QueuedAction
) -> None:
    """Test that release calls Redis delete."""
    await guard.release(sample_action)

    mock_redis.delete.assert_called_once()
    lock_key = guard._lock_key(sample_action.idempotency_key)
    assert mock_redis.delete.call_args[0][0] == lock_key


@pytest.mark.asyncio
async def test_release_is_idempotent_when_key_already_gone(
    guard: IdempotencyGuard, mock_redis: AsyncMock, sample_action: QueuedAction
) -> None:
    """Test that release doesn't fail if key is already deleted."""
    # Simulate Redis error when deleting non-existent key
    mock_redis.delete.side_effect = Exception("key not found")

    # Should not raise
    await guard.release(sample_action)

    mock_redis.delete.assert_called_once()
