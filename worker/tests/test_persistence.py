"""Unit tests for JobsRepository (Step 6)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from shared.contracts import QueuedAction

from app.db.models import WorkerActionJob
from app.db.repository import JobsRepository


@pytest.fixture
def mock_session_factory():
    """Mock async sessionmaker (sync callable returning async context manager)."""
    mock_session = AsyncMock()
    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_session
    mock_session_factory.return_value.__aexit__.return_value = None
    # Attach the session for easy access in tests
    mock_session_factory._mock_session = mock_session
    return mock_session_factory


@pytest.fixture
def jobs_repo(mock_session_factory):
    """JobsRepository with mocked session factory."""
    return JobsRepository(mock_session_factory)


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
async def test_create_or_get_job_inserts_new_row(jobs_repo, sample_action, mock_session_factory):
    """Test that create_or_get_job inserts a new row if idempotency_key is missing."""
    mock_session = mock_session_factory._mock_session
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()

    # Mock the second execute (SELECT) to return a job
    job_row = WorkerActionJob(
        idempotency_key=sample_action.idempotency_key,
        investigation_id=sample_action.investigation_id,
        model_name=sample_action.model_name,
        action_type=sample_action.action_type,
        target_version=sample_action.target_version,
        status="pending",
        attempt=0,
        max_attempts=3,
        payload=sample_action.model_dump(),
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    # First call is INSERT, second call is SELECT
    mock_session.execute.side_effect = [
        AsyncMock(),  # INSERT result
        MagicMock(scalar_one=MagicMock(return_value=job_row)),  # SELECT result
    ]

    job = await jobs_repo.create_or_get_job(sample_action)

    assert job.idempotency_key == sample_action.idempotency_key
    assert job.status == "pending"
    assert mock_session.execute.call_count == 2
    assert mock_session.commit.call_count == 1


@pytest.mark.asyncio
async def test_mark_running_updates_status(jobs_repo, mock_session_factory):
    """Test that mark_running sets status to 'running'."""
    mock_session = mock_session_factory._mock_session
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()

    await jobs_repo.mark_running("test-key", 1)

    mock_session.execute.assert_called_once()
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_mark_succeeded_sets_status(jobs_repo, mock_session_factory):
    """Test that mark_succeeded sets status to 'succeeded'."""
    mock_session = mock_session_factory._mock_session
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()

    await jobs_repo.mark_succeeded("test-key", result={"status": "ok"})

    mock_session.execute.assert_called_once()
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_mark_dead_lettered_sets_status(jobs_repo, mock_session_factory):
    """Test that mark_dead_lettered sets status and error."""
    mock_session = mock_session_factory._mock_session
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()

    await jobs_repo.mark_dead_lettered("test-key", "test error")

    mock_session.execute.assert_called_once()
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_mark_retry_scheduled_increments_attempt(jobs_repo, mock_session_factory):
    """Test that mark_retry_scheduled updates status and attempt."""
    mock_session = mock_session_factory._mock_session
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()

    await jobs_repo.mark_retry_scheduled("test-key", 1)

    mock_session.execute.assert_called_once()
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_mark_blocked_sets_blocked_status(jobs_repo, mock_session_factory):
    """Test that mark_blocked sets status to 'blocked'."""
    mock_session = mock_session_factory._mock_session
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()

    await jobs_repo.mark_blocked("test-key", "handler blocked")

    mock_session.execute.assert_called_once()
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_get_status_returns_current_status(jobs_repo, mock_session_factory):
    """Test that get_status retrieves the current status."""
    mock_session = mock_session_factory._mock_session

    # Mock execute to return a result object with scalar() method
    mock_result = MagicMock()
    mock_result.scalar.return_value = "succeeded"
    mock_session.execute = AsyncMock(return_value=mock_result)

    status = await jobs_repo.get_status("test-key")

    assert status == "succeeded"
    mock_session.execute.assert_called_once()
