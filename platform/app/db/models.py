"""ORM models for the platform service.

Currently one table — ``predictions_log`` — used by the drift scheduler
(future step) to read recent rows.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    Index,
    SmallInteger,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class PredictionLog(Base):
    """One row per ``POST /predict`` response."""

    __tablename__ = "predictions_log"
    __table_args__ = (
        CheckConstraint("label in (0, 1)", name="ck_predictions_log_label_binary"),
        Index("ix_predictions_log_predicted_at_desc", "predicted_at", postgresql_using="btree"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    features: Mapped[dict] = mapped_column(JSON, nullable=False)
