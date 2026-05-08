"""Repository for worker action jobs persistence."""

from __future__ import annotations

import structlog
from shared.contracts import QueuedAction
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import WorkerActionJob

log = structlog.get_logger(__name__)


class JobsRepository:
    """Manages persistence of action job lifecycle."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def create_or_get_job(self, action: QueuedAction) -> WorkerActionJob:
        """Insert a new job if it doesn't exist, or return the existing row.

        This method:
        - Inserts a new row if idempotency_key is missing (status='pending')
        - Returns the existing row if already present (does not overwrite terminal statuses)
        - Never resets succeeded/dead_lettered/blocked jobs back to pending

        Used on every claim and on retry bump to ensure row exists.
        """
        async with self.session_factory() as session:
            # Use ON CONFLICT to handle race conditions
            stmt = pg_insert(WorkerActionJob).values(
                idempotency_key=action.idempotency_key,
                investigation_id=action.investigation_id,
                model_name=action.model_name,
                action_type=action.action_type,
                target_version=action.target_version,
                status="pending",
                attempt=action.attempt,
                max_attempts=action.max_attempts,
                payload=action.model_dump(),
            )
            # On conflict, do nothing (preserve existing row/status)
            stmt = stmt.on_conflict_do_nothing(index_elements=["idempotency_key"])
            await session.execute(stmt)
            await session.commit()

            # Fetch and return the row (either newly inserted or existing)
            result = await session.execute(
                select(WorkerActionJob).where(
                    WorkerActionJob.idempotency_key == action.idempotency_key
                )
            )
            job = result.scalar_one()
            return job

    async def mark_running(self, idempotency_key: str, attempt: int) -> None:
        """Mark a job as running (in progress)."""
        from datetime import UTC, datetime

        async with self.session_factory() as session:
            stmt = (
                update(WorkerActionJob)
                .where(WorkerActionJob.idempotency_key == idempotency_key)
                .values(
                    status="running",
                    attempt=attempt,
                    started_at=datetime.now(UTC),
                )
            )
            await session.execute(stmt)
            await session.commit()

    async def mark_succeeded(self, idempotency_key: str, result: dict | None = None) -> None:
        """Mark a job as succeeded."""
        async with self.session_factory() as session:
            from datetime import UTC, datetime

            stmt = (
                update(WorkerActionJob)
                .where(WorkerActionJob.idempotency_key == idempotency_key)
                .values(
                    status="succeeded",
                    result=result,
                    finished_at=datetime.now(UTC),
                    last_error=None,
                )
            )
            await session.execute(stmt)
            await session.commit()

    async def mark_dead_lettered(self, idempotency_key: str, last_error: str | None = None) -> None:
        """Mark a job as dead-lettered (unrecoverable)."""
        async with self.session_factory() as session:
            from datetime import UTC, datetime

            stmt = (
                update(WorkerActionJob)
                .where(WorkerActionJob.idempotency_key == idempotency_key)
                .values(
                    status="dead_lettered",
                    last_error=last_error,
                    finished_at=datetime.now(UTC),
                )
            )
            await session.execute(stmt)
            await session.commit()

    async def mark_retry_scheduled(self, idempotency_key: str, attempt: int) -> None:
        """Mark a job as having a retry scheduled."""
        async with self.session_factory() as session:
            stmt = (
                update(WorkerActionJob)
                .where(WorkerActionJob.idempotency_key == idempotency_key)
                .values(
                    status="retry_scheduled",
                    attempt=attempt,
                )
            )
            await session.execute(stmt)
            await session.commit()

    async def mark_blocked(self, idempotency_key: str, last_error: str | None = None) -> None:
        """Mark a job as blocked (cannot proceed without manual intervention)."""
        async with self.session_factory() as session:
            from datetime import UTC, datetime

            stmt = (
                update(WorkerActionJob)
                .where(WorkerActionJob.idempotency_key == idempotency_key)
                .values(
                    status="blocked",
                    last_error=last_error,
                    finished_at=datetime.now(UTC),
                )
            )
            await session.execute(stmt)
            await session.commit()

    async def mark_skipped_duplicate(self, idempotency_key: str) -> None:
        """Mark a job as skipped because it already succeeded."""
        async with self.session_factory() as session:
            stmt = (
                update(WorkerActionJob)
                .where(WorkerActionJob.idempotency_key == idempotency_key)
                .values(
                    status="skipped_duplicate",
                )
            )
            await session.execute(stmt)
            await session.commit()

    async def get_status(self, idempotency_key: str) -> str | None:
        """Get the current status of a job by idempotency key."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(WorkerActionJob.status).where(
                    WorkerActionJob.idempotency_key == idempotency_key
                )
            )
            status = result.scalar()
            return status
