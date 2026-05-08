"""Idempotency guard using Redis SETNX lock + Postgres cached success lookup (Step 7)."""

from __future__ import annotations

from enum import StrEnum

import structlog
from redis.asyncio import Redis
from shared.contracts import QueuedAction

from app.db.repository import JobsRepository

log = structlog.get_logger(__name__)


class GuardOutcome(StrEnum):
    """Result of an idempotency guard acquisition."""

    PROCEED = "proceed"  # Safe to dispatch
    CACHED_SUCCESS = "cached_success"  # Already succeeded; skip dispatch
    LOCK_BUSY = "lock_busy"  # Another worker mid-flight; requeue


class IdempotencyGuard:
    """Two-layer idempotency: Postgres cache + Redis SETNX lock.

    - Postgres check: if status='succeeded', return CACHED_SUCCESS
    - Redis lock: if SETNX succeeds, return PROCEED; else LOCK_BUSY
    """

    LOCK_TTL_SECONDS: int = 300
    LOCK_PREFIX: str = "worker:lock:"

    def __init__(self, redis_client: Redis, jobs: JobsRepository) -> None:
        self.redis = redis_client
        self.jobs = jobs

    def _lock_key(self, idempotency_key: str) -> str:
        """Construct the Redis lock key."""
        return f"{self.LOCK_PREFIX}{idempotency_key}"

    async def acquire(self, action: QueuedAction) -> GuardOutcome:
        """Check for cached success and attempt to acquire a lock.

        1. Check Postgres for status='succeeded' → CACHED_SUCCESS
        2. Try Redis SETNX with TTL → PROCEED or LOCK_BUSY
        """
        # Check Postgres for cached success (terminal state)
        status = await self.jobs.get_status(action.idempotency_key)
        if status == "succeeded":
            log.info("idempotency.cached_success", key=action.idempotency_key)
            return GuardOutcome.CACHED_SUCCESS

        # Try to acquire the Redis lock
        lock_key = self._lock_key(action.idempotency_key)
        lock_acquired = await self.redis.set(
            lock_key,
            "1",
            nx=True,  # Only set if not exists
            ex=self.LOCK_TTL_SECONDS,
        )

        if lock_acquired:
            log.debug("idempotency.lock_acquired", key=action.idempotency_key)
            return GuardOutcome.PROCEED
        else:
            log.warning("idempotency.lock_busy", key=action.idempotency_key)
            return GuardOutcome.LOCK_BUSY

    async def release(self, action: QueuedAction) -> None:
        """Release the Redis lock (idempotent — no error if gone)."""
        lock_key = self._lock_key(action.idempotency_key)
        try:
            await self.redis.delete(lock_key)
            log.debug("idempotency.lock_released", key=action.idempotency_key)
        except Exception as exc:
            # Don't fail if the lock is already gone
            log.warning(
                "idempotency.lock_release_failed",
                key=action.idempotency_key,
                error=str(exc),
            )
