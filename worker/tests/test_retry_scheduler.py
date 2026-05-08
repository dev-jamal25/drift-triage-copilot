"""RetryScheduler tests — Step 5.

Pure unit tests with a mocked Redis client. The scheduler's job is one
EVAL call per tick, with the right keys and the current epoch.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.queue.keys import READY, RETRY
from app.runtime.retry_scheduler import RetryScheduler


@pytest.fixture
def client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def scheduler(client: AsyncMock) -> RetryScheduler:
    return RetryScheduler(client, tick_interval_seconds=0.01)


async def test_tick_evals_with_retry_and_ready_keys(
    client: AsyncMock,
    scheduler: RetryScheduler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each tick: one EVAL with KEYS=[RETRY, READY] and ARGV=[now]."""
    monkeypatch.setattr("app.runtime.retry_scheduler._now_epoch", lambda: 12345.0)
    client.eval.return_value = 0

    moved = await scheduler.tick()

    assert moved == 0
    client.eval.assert_awaited_once()
    args, _ = client.eval.await_args
    # Positional: (script, numkeys, key1, key2, argv1)
    script, numkeys, key1, key2, now_arg = args
    assert "ZRANGEBYSCORE" in script
    assert "LPUSH" in script
    assert "ZREM" in script
    assert numkeys == 2
    assert key1 == RETRY
    assert key2 == READY
    assert now_arg == 12345.0


async def test_tick_returns_int_count_promoted(
    client: AsyncMock, scheduler: RetryScheduler
) -> None:
    """The scheduler coerces the EVAL response to int."""
    client.eval.return_value = 7

    moved = await scheduler.tick()

    assert moved == 7
    assert isinstance(moved, int)


async def test_run_exits_when_shutdown_set(client: AsyncMock, scheduler: RetryScheduler) -> None:
    """Setting the event before run() means the loop exits without
    issuing any tick."""
    client.eval.return_value = 0
    shutdown = asyncio.Event()
    shutdown.set()

    await scheduler.run(shutdown)

    client.eval.assert_not_called()


async def test_run_swallows_redis_errors_and_continues(
    client: AsyncMock, scheduler: RetryScheduler
) -> None:
    """A transient ``RedisError`` on tick must NOT kill the scheduler;
    the next tick proceeds normally."""
    shutdown = asyncio.Event()
    call_count = 0

    async def flaky_eval(*_args: object, **_kwargs: object) -> int:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RedisConnectionError("transient")
        # Second tick: signal shutdown so run() exits quickly.
        shutdown.set()
        return 0

    client.eval.side_effect = flaky_eval

    await scheduler.run(shutdown)

    assert call_count >= 2  # we kept ticking after the error


async def test_run_does_not_swallow_programming_errors(
    client: AsyncMock, scheduler: RetryScheduler
) -> None:
    """A non-Redis exception (e.g. a bug) must propagate so we notice."""
    client.eval.side_effect = RuntimeError("bug in scheduler")
    shutdown = asyncio.Event()

    with pytest.raises(RuntimeError, match="bug in scheduler"):
        await scheduler.run(shutdown)
