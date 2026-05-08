"""create predictions_log

Revision ID: 0001
Revises:
Create Date: 2026-05-07 00:00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # gen_random_uuid() lives in pgcrypto on Postgres < 13; harmless on >= 13.
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "predictions_log",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("predicted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("label", sa.SmallInteger(), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.CheckConstraint("label in (0, 1)", name="ck_predictions_log_label_binary"),
    )
    op.create_index(
        "ix_predictions_log_predicted_at_desc",
        "predictions_log",
        ["predicted_at"],
        postgresql_using="btree",
    )


def downgrade() -> None:
    op.drop_index("ix_predictions_log_predicted_at_desc", table_name="predictions_log")
    op.drop_table("predictions_log")
