"""Retry scheduler — promotes due ``:retry`` entries to ``:ready``.

Runs as a separate concurrent task alongside the main claim/dispatch
loop. Every ``tick_interval_seconds`` it asks Redis to atomically
move all entries with ``score <= now`` from ``worker:queue:retry`` to
``worker:queue:ready``.

The promotion is atomic per script invocation via ``EVAL``. Without
atomicity, a crash between ``LPUSH :ready`` and ``ZREM :retry`` could
leave duplicates on ``:ready`` — Step 7's idempotency layer would
de-dupe via the Postgres unique key, but louder than necessary. The
Lua script eliminates that window entirely.
"""

from __future__ import annotations

import asyncio

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.queue.adapter import _now_epoch
from app.queue.keys import READY, RETRY

log = structlog.get_logger(__name__)

DEFAULT_TICK_INTERVAL_SECONDS: float = 1.0

# KEYS[1] = retry zset, KEYS[2] = ready list. ARGV[1] = now epoch (string).
# Returns the number of entries promoted.
_PROMOTE_DUE_SCRIPT = """
local entries = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
for i, entry in ipairs(entries) do
    redis.call('LPUSH', KEYS[2], entry)
    redis.call('ZREM', KEYS[1], entry)
end
return #entries
"""


class RetryScheduler:
    """Promotes due ``:retry`` entries to ``:ready`` on a 1Hz cadence."""

    def __init__(
        self,
        client: Redis,
        *,
        tick_interval_seconds: float = DEFAULT_TICK_INTERVAL_SECONDS,
    ) -> None:
        self._client = client
        self._tick_interval = tick_interval_seconds

    async def tick(self) -> int:
        """Promote all entries due now. Returns the count promoted."""
        now = _now_epoch()
        moved = await self._client.eval(_PROMOTE_DUE_SCRIPT, 2, RETRY, READY, now)
        count = int(moved)
        if count:
            log.info("retry_scheduler.promoted", count=count)
        return count

    async def run(self, shutdown: asyncio.Event) -> None:
        """Run the tick loop until ``shutdown`` is set.

        Catches ``RedisError`` per-tick so a transient Redis failure
        doesn't kill the scheduler task. Programming errors (Lua bugs,
        attribute errors) propagate as before.
        """
        log.info("retry_scheduler.start", tick_interval_seconds=self._tick_interval)
        while not shutdown.is_set():
            try:
                await self.tick()
            except RedisError:
                log.exception("retry_scheduler.tick_failed")
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=self._tick_interval)
            except TimeoutError:
                continue
        log.info("retry_scheduler.stop")
