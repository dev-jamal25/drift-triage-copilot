"""Retry policy + loop integration — Step 5.

Two layers of tests:
  - ``RetryPolicy.delay_for_attempt`` — formula and capping.
  - ``process_message`` with ``HandlerResult.retryable_failure`` —
    re-enqueue with bumped ``attempt``, or DLQ when ``max_attempts``
    is exhausted.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from shared.contracts import QueuedAction

from app.handlers.base import HandlerResult
from app.queue.adapter import RedisQueueAdapter
from app.runtime.loop import process_message
from app.runtime.retry_policy import RetryPolicy

# ----- RetryPolicy unit tests -----


def test_delay_for_attempt_zero_is_base() -> None:
    policy = RetryPolicy(backoff_base_seconds=2.0, retry_max_backoff_seconds=60.0)
    assert policy.delay_for_attempt(0) == 2.0


def test_delay_for_attempt_doubles_each_step() -> None:
    """``delay = base * 2**attempt``: 2, 4, 8, 16, 32 …"""
    policy = RetryPolicy(backoff_base_seconds=2.0, retry_max_backoff_seconds=1000.0)
    assert policy.delay_for_attempt(0) == 2.0
    assert policy.delay_for_attempt(1) == 4.0
    assert policy.delay_for_attempt(2) == 8.0
    assert policy.delay_for_attempt(3) == 16.0
    assert policy.delay_for_attempt(4) == 32.0


def test_delay_caps_at_max_backoff() -> None:
    policy = RetryPolicy(backoff_base_seconds=2.0, retry_max_backoff_seconds=10.0)
    # 2 * 2**3 = 16 → capped at 10
    assert policy.delay_for_attempt(3) == 10.0
    # Very large attempt — must still cap, not blow up.
    assert policy.delay_for_attempt(100) == 10.0


def test_delay_with_non_default_base() -> None:
    policy = RetryPolicy(backoff_base_seconds=5.0, retry_max_backoff_seconds=100.0)
    assert policy.delay_for_attempt(0) == 5.0
    assert policy.delay_for_attempt(2) == 20.0


# ----- Loop integration with retryable_failure -----


def _action(attempt: int, max_attempts: int = 3) -> QueuedAction:
    return QueuedAction(
        idempotency_key="inv1:replay_test:v3",
        investigation_id="inv1",
        model_name="bank-marketing-classifier",
        action_type="replay_test",
        target_version="v3",
        payload={"foo": "bar"},
        attempt=attempt,
        max_attempts=max_attempts,
        created_at=datetime(2026, 5, 7, tzinfo=UTC),
    )


def _payload_for(action: QueuedAction) -> bytes:
    return action.model_dump_json().encode("utf-8")


@pytest.fixture
def adapter() -> AsyncMock:
    return AsyncMock(spec=RedisQueueAdapter)


@pytest.fixture
def retry_policy() -> RetryPolicy:
    return RetryPolicy(backoff_base_seconds=2.0, retry_max_backoff_seconds=60.0)


def _retryable(error_type: str = "TransientError") -> AsyncMock:
    return AsyncMock(
        return_value=HandlerResult(
            status="retryable_failure",
            error_type=error_type,
            error_msg="simulated transient failure",
        )
    )


async def test_retryable_failure_first_attempt_nacks_with_bumped_attempt(
    adapter: AsyncMock, retry_policy: RetryPolicy
) -> None:
    """attempt=0, max_attempts=3 → next_attempt=1 < 3 → nack_retry with
    delay = 2 * 2**0 = 2.0 and a retry_raw whose ``attempt`` field is 1."""
    action = _action(attempt=0)
    raw = _payload_for(action)

    await process_message(adapter, raw, dispatch_fn=_retryable(), retry_policy=retry_policy)

    adapter.ack.assert_not_called()
    adapter.dead_letter.assert_not_called()
    adapter.nack_retry.assert_awaited_once()

    args, kwargs = adapter.nack_retry.await_args
    assert args[0] == raw  # the original in-flight bytes (for LREM)
    assert args[1] == 2.0  # delay = base * 2**0
    retry_raw = kwargs["retry_raw"]
    bumped = json.loads(retry_raw)
    assert bumped["attempt"] == 1
    # Other fields are preserved verbatim.
    assert bumped["idempotency_key"] == "inv1:replay_test:v3"
    assert bumped["payload"] == {"foo": "bar"}


async def test_retryable_failure_second_attempt_uses_bigger_delay(
    adapter: AsyncMock, retry_policy: RetryPolicy
) -> None:
    """attempt=1 → delay = 2 * 2**1 = 4.0; bumped attempt = 2."""
    action = _action(attempt=1)
    raw = _payload_for(action)

    await process_message(adapter, raw, dispatch_fn=_retryable(), retry_policy=retry_policy)

    args, kwargs = adapter.nack_retry.await_args
    assert args[1] == 4.0
    bumped = json.loads(kwargs["retry_raw"])
    assert bumped["attempt"] == 2


async def test_retryable_failure_at_last_attempt_dlqs_with_max_exhausted(
    adapter: AsyncMock, retry_policy: RetryPolicy
) -> None:
    """attempt=2, max_attempts=3 → next_attempt=3, NOT < 3 → DLQ
    ``max_attempts_exhausted``. error_type carries the handler's tag."""
    action = _action(attempt=2, max_attempts=3)
    raw = _payload_for(action)

    await process_message(
        adapter,
        raw,
        dispatch_fn=_retryable(error_type="UpstreamTimeout"),
        retry_policy=retry_policy,
    )

    adapter.ack.assert_not_called()
    adapter.nack_retry.assert_not_called()
    adapter.dead_letter.assert_awaited_once_with(
        raw,
        reason="max_attempts_exhausted",
        error_type="UpstreamTimeout",
        attempt=2,
    )


async def test_retryable_failure_with_max_attempts_one_dlqs_immediately(
    adapter: AsyncMock, retry_policy: RetryPolicy
) -> None:
    """``max_attempts=1`` means a single try; the first failure DLQs."""
    action = _action(attempt=0, max_attempts=1)
    raw = _payload_for(action)

    await process_message(adapter, raw, dispatch_fn=_retryable(), retry_policy=retry_policy)

    adapter.nack_retry.assert_not_called()
    adapter.dead_letter.assert_awaited_once()
    kwargs = adapter.dead_letter.await_args.kwargs
    assert kwargs["reason"] == "max_attempts_exhausted"
    assert kwargs["attempt"] == 0


async def test_retryable_failure_uses_capped_delay_at_high_attempt(
    adapter: AsyncMock,
) -> None:
    """A small ``retry_max_backoff_seconds`` caps the geometric
    progression — 2 * 2**3 = 16 → capped at 5."""
    tight_policy = RetryPolicy(backoff_base_seconds=2.0, retry_max_backoff_seconds=5.0)
    action = _action(attempt=3, max_attempts=10)
    raw = _payload_for(action)

    await process_message(adapter, raw, dispatch_fn=_retryable(), retry_policy=tight_policy)

    args, _ = adapter.nack_retry.await_args
    assert args[1] == 5.0


async def test_retryable_then_terminal_paths_do_not_double_handle(
    adapter: AsyncMock, retry_policy: RetryPolicy
) -> None:
    """Sanity: a single retryable_failure produces exactly one nack_retry
    and zero ack/dead_letter calls."""
    action = _action(attempt=0)
    raw = _payload_for(action)

    await process_message(adapter, raw, dispatch_fn=_retryable(), retry_policy=retry_policy)

    assert adapter.ack.call_count == 0
    assert adapter.dead_letter.call_count == 0
    assert adapter.nack_retry.call_count == 1
