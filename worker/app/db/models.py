"""Worker ORM models."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import CheckConstraint, DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class WorkerActionJob(Base):
    """Record of a queued action's lifecycle."""

    __tablename__ = "worker_action_jobs"

    idempotency_key: Mapped[str] = mapped_column(String(255), primary_key=True)
    investigation_id: Mapped[str] = mapped_column(String(255), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    action_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    attempt: Mapped[int] = mapped_column(default=0)
    max_attempts: Mapped[int] = mapped_column(default=3)
    last_error: Mapped[str | None] = mapped_column(Text(), nullable=True, default=None)
    result: Mapped[dict | None] = mapped_column(JSONB(), nullable=True, default=None)
    payload: Mapped[dict] = mapped_column(JSONB(), nullable=False, default={})
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        CheckConstraint(
            "status in ('pending', 'running', 'succeeded', 'retry_scheduled', 'dead_lettered', 'blocked', 'skipped_duplicate')",
            name="ck_worker_action_jobs_status",
        ),
        CheckConstraint(
            "action_type in ('replay_test', 'retrain', 'rollback')",
            name="ck_worker_action_jobs_action_type",
        ),
    )
