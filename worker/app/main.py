"""Worker entry point.

Loads settings, configures structlog, opens the database and Redis clients,
builds the queue adapter, starts the retry scheduler concurrently with the
claim/dispatch loop, and shuts all down cleanly on SIGINT/SIGTERM.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from collections.abc import AsyncIterator

import structlog
from redis.asyncio import Redis

from app.core.config import WorkerSettings, redact_url
from app.core.logging import configure_logging
from app.db.engine import close_engine, open_engine
from app.db.repository import JobsRepository
from app.handlers.dispatcher import dispatch
from app.queue.adapter import RedisQueueAdapter
from app.runtime.idempotency import IdempotencyGuard
from app.runtime.loop import run as run_loop
from app.runtime.retry_policy import RetryPolicy
from app.runtime.retry_scheduler import RetryScheduler

log = structlog.get_logger(__name__)


@contextlib.asynccontextmanager
async def open_redis(redis_url: str) -> AsyncIterator[Redis]:
    """Open a Redis client and ensure it is closed on exit."""
    client: Redis = Redis.from_url(redis_url, decode_responses=False)
    try:
        await client.ping()
        log.info("redis_client_open", url=redact_url(redis_url))
        yield client
    finally:
        await client.aclose()
        log.info("redis_client_closed")


def _install_shutdown_handlers(shutdown: asyncio.Event) -> None:
    """Trip ``shutdown`` on SIGINT/SIGTERM where the platform supports it."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown.set)
        except NotImplementedError:
            # Windows event loops don't implement add_signal_handler. Fall back
            # to the default handlers; KeyboardInterrupt still exits the loop.
            log.debug("signal_handler_unavailable", signal=sig.name)


async def main() -> None:
    settings = WorkerSettings()
    configure_logging(settings.log_level)

    log.info(
        "worker.start",
        log_level=settings.log_level,
        database_url=redact_url(settings.database_url),
        redis_url=redact_url(settings.redis_url),
        model_service_url=settings.model_service_url,
        max_retries=settings.max_retries,
        backoff_base_seconds=settings.backoff_base_seconds,
        retry_max_backoff_seconds=settings.retry_max_backoff_seconds,
        http_timeout_seconds=settings.http_timeout_seconds,
    )

    shutdown = asyncio.Event()
    _install_shutdown_handlers(shutdown)

    retry_policy = RetryPolicy(
        backoff_base_seconds=settings.backoff_base_seconds,
        retry_max_backoff_seconds=settings.retry_max_backoff_seconds,
    )

    # Open database engine
    engine, session_factory = await open_engine(settings.database_url)
    jobs = JobsRepository(session_factory)

    try:
        async with open_redis(settings.redis_url) as redis_client:
            adapter = RedisQueueAdapter(redis_client)
            guard = IdempotencyGuard(redis_client, jobs)
            scheduler = RetryScheduler(redis_client)
            scheduler_task = asyncio.create_task(scheduler.run(shutdown), name="retry_scheduler")
            try:
                await run_loop(
                    adapter,
                    shutdown,
                    dispatch_fn=dispatch,
                    retry_policy=retry_policy,
                    jobs=jobs,
                    guard=guard,
                )
            finally:
                shutdown.set()
                await scheduler_task

        log.info("worker.stop")
    finally:
        await close_engine(engine)


if __name__ == "__main__":
    asyncio.run(main())
