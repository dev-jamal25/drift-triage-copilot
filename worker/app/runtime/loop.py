"""Worker claim/validate/dispatch loop — Steps 5–7.

Pulls one message at a time, validates against
``shared.contracts.QueuedAction``, checks idempotency (Postgres cache + Redis lock),
persists job lifecycle to Postgres, dispatches to a handler, and routes the outcome:

  - ``ValidationError``         → DLQ ``validation_error``  (no retry)
  - Idempotency cached success  → ack (skip dispatch)
  - Idempotency lock busy       → requeue with short delay (handler not run)
  - ``RollbackBlockedError``    → mark blocked, DLQ ``rollback_blocked_no_promotion_gate``
  - ``HandlerResult.success``   → mark succeeded, ack
  - ``HandlerResult.terminal_failure`` → mark dead_lettered, DLQ
  - ``HandlerResult.retryable_failure`` →
        if next_attempt < max_attempts: mark retry_scheduled, ``nack_retry`` with bumped attempt;
        else: mark dead_lettered, DLQ.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import structlog
from pydantic import ValidationError
from shared.contracts import QueuedAction

from app.db.repository import JobsRepository
from app.handlers.base import HandlerResult, RollbackBlockedError
from app.queue.adapter import RedisQueueAdapter
from app.runtime.idempotency import GuardOutcome, IdempotencyGuard
from app.runtime.retry_policy import RetryPolicy

log = structlog.get_logger(__name__)

DEFAULT_CLAIM_TIMEOUT_SECONDS: float = 5.0

DispatchFn = Callable[[QueuedAction], Awaitable[HandlerResult]]


async def process_message(
    adapter: RedisQueueAdapter,
    raw: bytes,
    *,
    dispatch_fn: DispatchFn,
    retry_policy: RetryPolicy,
    jobs: JobsRepository,
    guard: IdempotencyGuard,
) -> None:
    """Validate, dispatch, and route one raw message. Check idempotency, persist status."""
    try:
        action = QueuedAction.model_validate_json(raw)
    except ValidationError as exc:
        log.error(
            "queue.validation_failed",
            error_count=len(exc.errors()),
            error_summary=str(exc)[:500],
        )
        await adapter.dead_letter(
            raw,
            reason="validation_error",
            error_type="ValidationError",
            attempt=0,
        )
        return

    log.info(
        "queue.action_validated",
        idempotency_key=action.idempotency_key,
        action_type=action.action_type,
        investigation_id=action.investigation_id,
        target_version=action.target_version,
        attempt=action.attempt,
    )

    # Check idempotency: cached success or lock busy
    outcome = await guard.acquire(action)
    if outcome is GuardOutcome.CACHED_SUCCESS:
        log.info("idempotency.cached_success", key=action.idempotency_key)
        await adapter.ack(raw)
        return
    if outcome is GuardOutcome.LOCK_BUSY:
        log.warning("idempotency.lock_busy", key=action.idempotency_key)
        # Requeue with short delay; handler did not run, so no attempt increment
        await adapter.nack_retry(raw, delay_seconds=5.0)
        return

    # outcome == PROCEED — continue with dispatch under lock protection
    # Persist job record (create if missing)
    await jobs.create_or_get_job(action)
    # Mark as running
    await jobs.mark_running(action.idempotency_key, action.attempt)

    try:
        try:
            result = await dispatch_fn(action)
        except RollbackBlockedError as exc:
            log.error(
                "handler.rollback.dlq",
                idempotency_key=action.idempotency_key,
                error_msg=str(exc),
            )
            await jobs.mark_blocked(action.idempotency_key, str(exc))
            await adapter.dead_letter(
                raw,
                reason="rollback_blocked_no_promotion_gate",
                error_type="RollbackBlockedError",
                attempt=action.attempt,
            )
            return

        if result.status == "success":
            await jobs.mark_succeeded(action.idempotency_key)
            await adapter.ack(raw)
            return

        if result.status == "terminal_failure":
            log.error(
                "handler.terminal_failure",
                idempotency_key=action.idempotency_key,
                action_type=action.action_type,
                error_type=result.error_type,
                error_msg=result.error_msg,
            )
            await jobs.mark_dead_lettered(action.idempotency_key, result.error_msg or "")
            await adapter.dead_letter(
                raw,
                reason="terminal_failure",
                error_type=result.error_type or "TerminalFailure",
                attempt=action.attempt,
            )
            return

        # retryable_failure
        next_attempt = action.attempt + 1
        if next_attempt >= action.max_attempts:
            log.error(
                "handler.max_attempts_exhausted",
                idempotency_key=action.idempotency_key,
                attempt=action.attempt,
                max_attempts=action.max_attempts,
                error_type=result.error_type,
                error_msg=result.error_msg,
            )
            await jobs.mark_dead_lettered(
                action.idempotency_key, result.error_msg or "max_attempts_exhausted"
            )
            await adapter.dead_letter(
                raw,
                reason="max_attempts_exhausted",
                error_type=result.error_type or "RetryableFailure",
                attempt=action.attempt,
            )
            return

        delay = retry_policy.delay_for_attempt(action.attempt)
        bumped = action.model_copy(update={"attempt": next_attempt})
        retry_raw = bumped.model_dump_json().encode("utf-8")
        log.warning(
            "handler.retryable_failure",
            idempotency_key=action.idempotency_key,
            attempt=action.attempt,
            next_attempt=next_attempt,
            delay_seconds=delay,
            error_type=result.error_type,
            error_msg=result.error_msg,
        )
        await jobs.mark_retry_scheduled(action.idempotency_key, next_attempt)
        await adapter.nack_retry(raw, delay, retry_raw=retry_raw)
    finally:
        # Always release the lock, even if an exception occurs
        await guard.release(action)


async def run(
    adapter: RedisQueueAdapter,
    shutdown: asyncio.Event,
    *,
    dispatch_fn: DispatchFn,
    retry_policy: RetryPolicy,
    jobs: JobsRepository,
    guard: IdempotencyGuard,
    claim_timeout_seconds: float = DEFAULT_CLAIM_TIMEOUT_SECONDS,
) -> None:
    """Run the claim/process loop until ``shutdown`` is set."""
    log.info("worker.loop.start", claim_timeout_seconds=claim_timeout_seconds)
    while not shutdown.is_set():
        raw = await adapter.claim(claim_timeout_seconds)
        if raw is None:
            continue
        await process_message(
            adapter,
            raw,
            dispatch_fn=dispatch_fn,
            retry_policy=retry_policy,
            jobs=jobs,
            guard=guard,
        )
    log.info("worker.loop.stop")
