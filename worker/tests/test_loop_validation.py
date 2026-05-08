"""Loop validation tests — Step 3 (carried into Step 4, updated for Step 6).

Pin down the validation path: malformed payloads go straight to the
DLQ and never reach the dispatcher.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from shared.contracts import QueuedAction

from app.db.repository import JobsRepository
from app.handlers.base import HandlerResult
from app.queue.adapter import RedisQueueAdapter
from app.runtime.idempotency import GuardOutcome, IdempotencyGuard
from app.runtime.loop import process_message
from app.runtime.retry_policy import RetryPolicy


def _valid_action() -> QueuedAction:
    return QueuedAction(
        idempotency_key="inv1:replay_test:v3",
        investigation_id="inv1",
        model_name="bank-marketing-classifier",
        action_type="replay_test",
        target_version="v3",
        payload={},
        attempt=0,
        max_attempts=3,
        created_at=datetime(2026, 5, 7, tzinfo=UTC),
    )


def _valid_payload() -> bytes:
    return _valid_action().model_dump_json().encode("utf-8")


@pytest.fixture
def adapter() -> AsyncMock:
    return AsyncMock(spec=RedisQueueAdapter)


@pytest.fixture
def dispatch_fn() -> AsyncMock:
    """Default dispatch_fn that returns ``HandlerResult(success)``."""
    fn = AsyncMock()
    fn.return_value = HandlerResult(status="success")
    return fn


@pytest.fixture
def retry_policy() -> RetryPolicy:
    """Production-default retry policy (matches WorkerSettings defaults)."""
    return RetryPolicy(backoff_base_seconds=2.0, retry_max_backoff_seconds=60.0)


@pytest.fixture
def jobs() -> AsyncMock:
    """Mock JobsRepository."""
    return AsyncMock(spec=JobsRepository)


@pytest.fixture
def guard() -> AsyncMock:
    """Mock IdempotencyGuard with PROCEED outcome."""
    mock = AsyncMock(spec=IdempotencyGuard)
    mock.acquire.return_value = GuardOutcome.PROCEED
    return mock


async def test_valid_payload_dispatches_then_acks(
    adapter: AsyncMock,
    dispatch_fn: AsyncMock,
    retry_policy: RetryPolicy,
    jobs: AsyncMock,
    guard: AsyncMock,
) -> None:
    raw = _valid_payload()

    await process_message(
        adapter, raw, dispatch_fn=dispatch_fn, retry_policy=retry_policy, jobs=jobs, guard=guard
    )

    dispatch_fn.assert_awaited_once()
    action = dispatch_fn.await_args.args[0]
    assert isinstance(action, QueuedAction)
    assert action.action_type == "replay_test"
    adapter.ack.assert_awaited_once_with(raw)
    adapter.dead_letter.assert_not_called()


async def test_malformed_json_goes_to_dlq(
    adapter: AsyncMock,
    dispatch_fn: AsyncMock,
    retry_policy: RetryPolicy,
    jobs: AsyncMock,
    guard: AsyncMock,
) -> None:
    raw = b"this is not json"

    await process_message(
        adapter, raw, dispatch_fn=dispatch_fn, retry_policy=retry_policy, jobs=jobs, guard=guard
    )

    dispatch_fn.assert_not_called()
    adapter.ack.assert_not_called()
    adapter.dead_letter.assert_awaited_once_with(
        raw,
        reason="validation_error",
        error_type="ValidationError",
        attempt=0,
    )
    # Validation failures don't reach idempotency guard or DB persistence
    guard.acquire.assert_not_called()
    jobs.create_or_get_job.assert_not_called()


async def test_missing_required_field_goes_to_dlq(
    adapter: AsyncMock,
    dispatch_fn: AsyncMock,
    retry_policy: RetryPolicy,
    jobs: AsyncMock,
    guard: AsyncMock,
) -> None:
    """``idempotency_key`` is required — omitting it must reject."""
    base = _valid_action().model_dump(mode="json")
    del base["idempotency_key"]
    raw = json.dumps(base).encode("utf-8")

    await process_message(
        adapter, raw, dispatch_fn=dispatch_fn, retry_policy=retry_policy, jobs=jobs, guard=guard
    )

    dispatch_fn.assert_not_called()
    adapter.ack.assert_not_called()
    kwargs = adapter.dead_letter.await_args.kwargs
    assert kwargs == {
        "reason": "validation_error",
        "error_type": "ValidationError",
        "attempt": 0,
    }
    guard.acquire.assert_not_called()
    jobs.create_or_get_job.assert_not_called()


async def test_extra_forbidden_field_goes_to_dlq(
    adapter: AsyncMock,
    dispatch_fn: AsyncMock,
    retry_policy: RetryPolicy,
    jobs: AsyncMock,
    guard: AsyncMock,
) -> None:
    """``QueuedAction`` has ``extra='forbid'`` — unknown fields reject."""
    base = _valid_action().model_dump(mode="json")
    base["unexpected"] = "boom"
    raw = json.dumps(base).encode("utf-8")

    await process_message(
        adapter, raw, dispatch_fn=dispatch_fn, retry_policy=retry_policy, jobs=jobs, guard=guard
    )

    dispatch_fn.assert_not_called()
    adapter.ack.assert_not_called()
    guard.acquire.assert_not_called()
    jobs.create_or_get_job.assert_not_called()
    adapter.dead_letter.assert_awaited_once()
    jobs.create_or_get_job.assert_not_called()


async def test_wrong_action_type_goes_to_dlq(
    adapter: AsyncMock,
    dispatch_fn: AsyncMock,
    retry_policy: RetryPolicy,
    jobs: AsyncMock,
    guard: AsyncMock,
) -> None:
    """``action_type`` is a Literal — anything outside the set rejects."""
    base = _valid_action().model_dump(mode="json")
    base["action_type"] = "BOGUS"
    raw = json.dumps(base).encode("utf-8")

    await process_message(
        adapter, raw, dispatch_fn=dispatch_fn, retry_policy=retry_policy, jobs=jobs, guard=guard
    )

    dispatch_fn.assert_not_called()
    adapter.ack.assert_not_called()
    adapter.dead_letter.assert_awaited_once()
    guard.acquire.assert_not_called()
    jobs.create_or_get_job.assert_not_called()


async def test_validation_failure_does_not_double_ack(
    adapter: AsyncMock,
    dispatch_fn: AsyncMock,
    retry_policy: RetryPolicy,
    jobs: AsyncMock,
    guard: AsyncMock,
) -> None:
    """An extra ack would re-LREM and could remove an unrelated message
    if the same bytes happen to occur twice in :processing. ``dead_letter``
    already issues the LREM."""
    await process_message(
        adapter, b"junk", dispatch_fn=dispatch_fn, retry_policy=retry_policy, jobs=jobs, guard=guard
    )

    adapter.ack.assert_not_called()
    adapter.dead_letter.assert_awaited_once()
    guard.acquire.assert_not_called()
    jobs.create_or_get_job.assert_not_called()


async def test_negative_attempt_goes_to_dlq(
    adapter: AsyncMock,
    dispatch_fn: AsyncMock,
    retry_policy: RetryPolicy,
    jobs: AsyncMock,
    guard: AsyncMock,
) -> None:
    """``attempt`` has ``ge=0`` — negative values reject."""
    base = _valid_action().model_dump(mode="json")
    base["attempt"] = -1
    raw = json.dumps(base).encode("utf-8")

    await process_message(
        adapter, raw, dispatch_fn=dispatch_fn, retry_policy=retry_policy, jobs=jobs, guard=guard
    )

    dispatch_fn.assert_not_called()
    adapter.dead_letter.assert_awaited_once()
    guard.acquire.assert_not_called()
    jobs.create_or_get_job.assert_not_called()
