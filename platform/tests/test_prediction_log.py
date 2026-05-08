"""Unit tests for the prediction-log service.

Mocks the AsyncSession — no Postgres or SQLite at runtime.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.db.models import PredictionLog
from app.services.prediction_log import fetch_recent, insert_prediction


@pytest.mark.asyncio
async def test_insert_prediction_adds_row_and_flushes() -> None:
    session = AsyncMock()
    session.add = MagicMock()  # session.add is sync; AsyncMock returns coroutine by default

    await insert_prediction(
        session,
        predicted_at=datetime(2026, 5, 7, 10, 30, tzinfo=UTC),
        model_name="bank-marketing-classifier",
        model_version="2",
        threshold=0.38,
        score=0.55,
        label=1,
        features={"age": 35, "job": "admin."},
    )

    session.add.assert_called_once()
    session.flush.assert_awaited_once()

    (added,) = session.add.call_args.args
    assert isinstance(added, PredictionLog)
    assert added.model_name == "bank-marketing-classifier"
    assert added.model_version == "2"
    assert added.threshold == 0.38
    assert added.score == 0.55
    assert added.label == 1
    assert added.features == {"age": 35, "job": "admin."}
    assert added.predicted_at == datetime(2026, 5, 7, 10, 30, tzinfo=UTC)


@pytest.mark.asyncio
async def test_fetch_recent_orders_desc_with_limit() -> None:
    session = AsyncMock()
    scalars = MagicMock()
    scalars.all.return_value = ["row1", "row2"]
    result = MagicMock()
    result.scalars.return_value = scalars
    session.execute.return_value = result

    rows = await fetch_recent(session, limit=42)

    session.execute.assert_awaited_once()
    stmt = session.execute.call_args.args[0]
    compiled = str(stmt)
    assert "FROM predictions_log" in compiled
    assert "ORDER BY predictions_log.predicted_at DESC" in compiled
    assert "LIMIT" in compiled
    assert rows == ["row1", "row2"]
