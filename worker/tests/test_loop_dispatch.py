"""Loop ↔ dispatcher integration — Step 4 + 5 (updated for Step 6).

Pure unit tests with a mocked queue adapter and the real dispatcher.
Covers the routing paths the loop owns:

  - ``HandlerResult.success`` from a stub handler → ack
  - ``HandlerResult.terminal_failure`` → DLQ ``terminal_failure``
  - ``RollbackBlockedError`` raised by rollback handler → DLQ
    ``rollback_blocked_no_promotion_gate``

Retry-specific routing (``retryable_failure``) is in
``test_retry_policy.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from shared.contracts import QueuedAction

from app.db.repository import JobsRepository
from app.handlers.base import HandlerResult
from app.handlers.dispatcher import dispatch
from app.queue.adapter import RedisQueueAdapter
from app.runtime.idempotency import GuardOutcome, IdempotencyGuard
from app.runtime.loop import process_message
from app.runtime.retry_policy import RetryPolicy


def _payload(action_type: str, *, attempt: int = 0) -> bytes:
    return (
        QueuedAction(
            idempotency_key=f"inv1:{action_type}:v3",
            investigation_id="inv1",
            model_name="bank-marketing-classifier",
            action_type=action_type,  # type: ignore[arg-type]
            target_version="v3",
            payload={},
            attempt=attempt,
            max_attempts=3,
            created_at=datetime(2026, 5, 7, tzinfo=UTC),
        )
        .model_dump_json()
        .encode("utf-8")
    )


@pytest.fixture
def adapter() -> AsyncMock:
    return AsyncMock(spec=RedisQueueAdapter)


@pytest.fixture
def retry_policy() -> RetryPolicy:
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


async def test_replay_test_action_acks(
    adapter: AsyncMock, retry_policy: RetryPolicy, jobs: AsyncMock, guard: AsyncMock
) -> None:
    raw = _payload("replay_test")

    await process_message(
        adapter, raw, dispatch_fn=dispatch, retry_policy=retry_policy, jobs=jobs, guard=guard
    )

    adapter.ack.assert_awaited_once_with(raw)
    adapter.dead_letter.assert_not_called()
    adapter.nack_retry.assert_not_called()


async def test_retrain_action_acks(
    adapter: AsyncMock, retry_policy: RetryPolicy, jobs: AsyncMock, guard: AsyncMock
) -> None:
    raw = _payload("retrain")

    await process_message(
        adapter, raw, dispatch_fn=dispatch, retry_policy=retry_policy, jobs=jobs, guard=guard
    )

    adapter.ack.assert_awaited_once_with(raw)
    adapter.dead_letter.assert_not_called()
    adapter.nack_retry.assert_not_called()


async def test_rollback_action_dlqs_with_blocked_reason(
    adapter: AsyncMock, retry_policy: RetryPolicy, jobs: AsyncMock, guard: AsyncMock
) -> None:
    raw = _payload("rollback", attempt=2)

    await process_message(
        adapter, raw, dispatch_fn=dispatch, retry_policy=retry_policy, jobs=jobs, guard=guard
    )

    adapter.ack.assert_not_called()
    adapter.nack_retry.assert_not_called()
    adapter.dead_letter.assert_awaited_once_with(
        raw,
        reason="rollback_blocked_no_promotion_gate",
        error_type="RollbackBlockedError",
        attempt=2,
    )


async def test_terminal_failure_from_dispatcher_dlqs(
    adapter: AsyncMock, retry_policy: RetryPolicy, jobs: AsyncMock, guard: AsyncMock
) -> None:
    """Inject a dispatch_fn that returns ``terminal_failure`` to exercise
    the loop's terminal-failure routing without depending on the
    Literal-blocked unknown-action path."""
    raw = _payload("replay_test", attempt=1)
    dispatch_fn = AsyncMock(
        return_value=HandlerResult(
            status="terminal_failure",
            error_type="ContrivedFailure",
            error_msg="for testing",
        )
    )

    await process_message(
        adapter, raw, dispatch_fn=dispatch_fn, retry_policy=retry_policy, jobs=jobs, guard=guard
    )

    adapter.ack.assert_not_called()
    adapter.nack_retry.assert_not_called()
    adapter.dead_letter.assert_awaited_once_with(
        raw,
        reason="terminal_failure",
        error_type="ContrivedFailure",
        attempt=1,
    )


async def test_unknown_action_terminal_failure_path_via_mocked_dispatch(
    adapter: AsyncMock, retry_policy: RetryPolicy, jobs: AsyncMock, guard: AsyncMock
) -> None:
    """End-to-end test for the path: dispatcher returns
    ``terminal_failure`` with ``error_type='UnknownActionType'`` →
    loop DLQs with that error_type. The Literal in QueuedAction blocks
    constructing this for real, so we mock the dispatcher's response."""
    raw = _payload("replay_test")
    dispatch_fn = AsyncMock(
        return_value=HandlerResult(
            status="terminal_failure",
            error_type="UnknownActionType",
            error_msg="no handler for 'mystery_action'",
        )
    )

    await process_message(
        adapter, raw, dispatch_fn=dispatch_fn, retry_policy=retry_policy, jobs=jobs, guard=guard
    )

    kwargs = adapter.dead_letter.await_args.kwargs
    assert kwargs["reason"] == "terminal_failure"
    assert kwargs["error_type"] == "UnknownActionType"
