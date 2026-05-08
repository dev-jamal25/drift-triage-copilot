"""worker_action_jobs timestamps to TIMESTAMPTZ

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-08 13:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_TZ_COLUMNS = ("created_at", "updated_at", "started_at", "finished_at")


def upgrade() -> None:
    for col in _TZ_COLUMNS:
        op.execute(
            f"ALTER TABLE worker_action_jobs "
            f"ALTER COLUMN {col} TYPE TIMESTAMP WITH TIME ZONE "
            f"USING {col} AT TIME ZONE 'UTC'"
        )


def downgrade() -> None:
    for col in _TZ_COLUMNS:
        op.execute(
            f"ALTER TABLE worker_action_jobs "
            f"ALTER COLUMN {col} TYPE TIMESTAMP WITHOUT TIME ZONE "
            f"USING {col} AT TIME ZONE 'UTC'"
        )
