"""Prediction-log persistence.

``insert_prediction`` is called from the request path. ``fetch_recent`` is
exposed now for the drift scheduler to consume in a later step; it's not
called from ``/predict``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import PredictionLog


async def insert_prediction(
    session: AsyncSession,
    *,
    predicted_at: datetime,
    model_name: str,
    model_version: str,
    threshold: float,
    score: float,
    label: int,
    features: dict[str, Any],
) -> None:
    """Persist a single prediction. Caller commits the session."""
    row = PredictionLog(
        predicted_at=predicted_at,
        model_name=model_name,
        model_version=model_version,
        threshold=threshold,
        score=score,
        label=label,
        features=features,
    )
    session.add(row)
    await session.flush()


async def fetch_recent(session: AsyncSession, limit: int) -> list[PredictionLog]:
    """Return the most recent ``limit`` predictions, newest first."""
    stmt = select(PredictionLog).order_by(PredictionLog.predicted_at.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())
