"""Reliable queue adapter over Redis primitives.

Topology:
  - ``worker:queue:ready``      LIST  (FIFO; producer = agent does LPUSH)
  - ``worker:queue:processing`` LIST  (in-flight; populated atomically)
  - ``worker:queue:retry``      ZSET  (score = unix epoch when re-eligible)
  - ``worker:queue:dlq``        LIST  (terminal; envelope-wrapped)

**Invariant:** every transition out of ``:processing`` copies to the
destination FIRST, then issues ``LREM :processing 1 raw``. Crashing
between the two leaves the message recoverable from ``:processing``
rather than lost. The downside — a brief duplicate during the window —
is acceptable under the single-worker-process invariant from
DECISIONS.md.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import structlog
from redis.asyncio import Redis

from app.queue.keys import DLQ, PROCESSING, READY, RETRY

log = structlog.get_logger(__name__)


def _now_epoch() -> float:
    """Unix-epoch wall clock; patched in tests for deterministic ZADD scores."""
    return time.time()


def _now_iso() -> str:
    """ISO-8601 UTC timestamp; patched in tests for deterministic envelopes."""
    return datetime.now(UTC).isoformat()


class RedisQueueAdapter:
    """Thin async wrapper around a single ``redis.asyncio.Redis`` client."""

    def __init__(self, client: Redis) -> None:
        self._client = client

    async def claim(self, timeout_seconds: float) -> bytes | None:
        """Atomically move one message from ``:ready`` to ``:processing``.

        Returns the raw bytes, or ``None`` on timeout.

        Producer convention: agent does ``LPUSH :ready raw`` (newest at
        the left). We pop the oldest (right) and push to the left of
        ``:processing`` so newer in-flight items sit closer to the head.
        """
        result = await self._client.blmove(
            READY, PROCESSING, timeout_seconds, src="RIGHT", dest="LEFT"
        )
        if result is None:
            return None
        return result if isinstance(result, bytes) else bytes(result)

    async def ack(self, raw: bytes) -> None:
        """Remove the in-flight message from ``:processing`` on success."""
        await self._client.lrem(PROCESSING, 1, raw)

    async def nack_retry(
        self,
        raw: bytes,
        delay_seconds: float,
        *,
        retry_raw: bytes | None = None,
    ) -> None:
        """Schedule the message for a future retry.

        ``retry_raw`` lets the caller pass an updated payload (e.g. with
        a bumped ``attempt`` field) that goes onto ``:retry``, while the
        *original* in-flight ``raw`` is the one removed from
        ``:processing``. Defaults to ``raw`` for the simple case.

        Order: ``ZADD :retry`` first, then ``LREM :processing`` — see
        the invariant at the top of this module.
        """
        to_retry = retry_raw if retry_raw is not None else raw
        score = _now_epoch() + delay_seconds
        await self._client.zadd(RETRY, {to_retry: score})
        await self._client.lrem(PROCESSING, 1, raw)

    async def dead_letter(
        self,
        raw: bytes,
        *,
        reason: str,
        error_type: str,
        attempt: int,
    ) -> None:
        """Move the message to the DLQ wrapped in a metadata envelope.

        Envelope shape: ``{raw, reason, error_type, failed_at, attempt}``.
        Order: ``LPUSH :dlq envelope`` first, then ``LREM :processing`` —
        see invariant. The LREM removes the *original* raw bytes, not the
        envelope.
        """
        envelope = json.dumps(
            {
                "raw": raw.decode("utf-8", errors="replace"),
                "reason": reason,
                "error_type": error_type,
                "failed_at": _now_iso(),
                "attempt": attempt,
            }
        ).encode("utf-8")
        await self._client.lpush(DLQ, envelope)
        await self._client.lrem(PROCESSING, 1, raw)
