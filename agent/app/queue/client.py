"""Redis queue abstraction for enqueuing actions."""

import os

import redis.asyncio as redis
import structlog
from shared.contracts import QueuedAction

log = structlog.get_logger()


class QueueClient:
    """Abstraction for enqueuing actions to the worker queue."""

    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://redis:6379")
        self.redis_client: redis.Redis | None = None

    async def connect(self) -> None:
        """Connect to Redis."""
        log.info("queue.connect", redis_url=self.redis_url)
        self.redis_client = await redis.from_url(self.redis_url, decode_responses=True)

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()

    async def enqueue(self, action: QueuedAction) -> bool:
        """
        Enqueue an action for the worker.

        Returns True if enqueued, False if already exists (idempotent).
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected. Call connect() first.")

        # Check idempotency
        idem_key = f"worker:queue:idem:{action.idempotency_key}"
        exists = await self.redis_client.exists(idem_key)

        if exists:
            log.warning("queue.enqueue.duplicate", idempotency_key=action.idempotency_key)
            return False

        # Enqueue the action
        queue_key = "worker:queue:ready"
        action_json = action.model_dump_json()

        try:
            await self.redis_client.rpush(queue_key, action_json)
            log.info(
                "queue.enqueue.success",
                idempotency_key=action.idempotency_key,
                investigation_id=action.investigation_id,
            )

            # Set idempotency key with 24-hour TTL
            await self.redis_client.setex(idem_key, 86400, "1")

            return True
        except Exception as e:
            log.error(
                "queue.enqueue.failed",
                idempotency_key=action.idempotency_key,
                error=str(e),
            )
            raise


# Global instance
_queue_client = None


def get_queue_client() -> QueueClient:
    """Get or create the global queue client."""
    global _queue_client
    if _queue_client is None:
        _queue_client = QueueClient()
    return _queue_client
