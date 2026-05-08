"""Unit tests for ``app.services.drift_service.recompute_drift``.

Mocks ``fetch_recent`` so no real DB is touched. The drift compute itself is
the Step 3 pure function — covered by ``test_drift_metrics.py`` — so here
we only assert the row-assembly and version-filter behaviour.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from app.services.drift_service import recompute_drift
from app.services.model_loader import ModelBundle
from ml.data import clean
from ml.drift_stats import compute_reference_stats


def _synthetic_raw(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 95, size=n),
            "job": rng.choice(["admin.", "blue-collar", "technician", "unknown"], size=n),
            "marital": rng.choice(["married", "single", "divorced", "unknown"], size=n),
            "education": rng.choice(
                ["basic.4y", "high.school", "university.degree", "unknown"], size=n
            ),
            "default": rng.choice(["no", "yes", "unknown"], size=n),
            "housing": rng.choice(["no", "yes", "unknown"], size=n),
            "loan": rng.choice(["no", "yes", "unknown"], size=n),
            "contact": rng.choice(["cellular", "telephone"], size=n),
            "month": rng.choice(["may", "jun", "jul", "aug"], size=n),
            "day_of_week": rng.choice(["mon", "tue", "wed", "thu", "fri"], size=n),
            "duration": rng.integers(0, 5000, size=n),
            "campaign": rng.integers(1, 10, size=n),
            "pdays": np.where(rng.random(n) < 0.7, 999, rng.integers(0, 30, size=n)),
            "previous": rng.integers(0, 5, size=n),
            "poutcome": rng.choice(["success", "failure", "nonexistent"], size=n),
            "emp.var.rate": rng.uniform(-3, 2, size=n),
            "cons.price.idx": rng.uniform(92, 95, size=n),
            "cons.conf.idx": rng.uniform(-50, -25, size=n),
            "euribor3m": rng.uniform(0.5, 5.0, size=n),
            "nr.employed": rng.uniform(4900, 5230, size=n),
            "y": rng.choice(["yes", "no"], size=n, p=[0.2, 0.8]),
        }
    )


def _ref_stats() -> dict[str, Any]:
    cleaned = clean(_synthetic_raw(n=600, seed=1))
    X = cleaned.drop(columns=["target"])
    y = cleaned["target"]
    proba = np.random.default_rng(9).random(len(X))
    return compute_reference_stats(X, proba, positive_rate=float(y.mean()))


def _bundle_with(stats: dict[str, Any] | None) -> ModelBundle:
    return ModelBundle(
        pipeline=MagicMock(),
        threshold=0.4,
        model_uri="models:/m@staging",
        model_name="m",
        version="1",
        alias="staging",
        run_id="r1",
        loaded_at=datetime.now(UTC),
        reference_stats=stats,
    )


def _row(version: str = "1", *, age: int = 35, score: float = 0.5, t_offset: int = 0) -> MagicMock:
    """Build a fake PredictionLog ORM row."""
    row = MagicMock()
    row.model_name = "m"
    row.model_version = version
    row.score = score
    row.predicted_at = datetime.now(UTC) - timedelta(seconds=t_offset)
    row.features = {
        "age": age,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "thu",
        "campaign": 2,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": -1.8,
        "cons.price.idx": 93.075,
        "cons.conf.idx": -47.1,
        "euribor3m": 1.405,
        "nr.employed": 5099.1,
    }
    return row


@pytest.mark.asyncio
async def test_recompute_drift_raises_when_reference_stats_missing() -> None:
    bundle = _bundle_with(stats=None)
    with pytest.raises(RuntimeError, match="reference_stats"):
        await recompute_drift(AsyncMock(), bundle, window_size=100)


@pytest.mark.asyncio
async def test_recompute_drift_returns_empty_report_when_no_matching_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle_with(stats=_ref_stats())

    async def fake_fetch(_session, _limit):
        # rows from a different model_version — should be filtered out
        return [_row(version="OTHER")]

    monkeypatch.setattr("app.services.drift_service.fetch_recent", fake_fetch)

    report = await recompute_drift(AsyncMock(), bundle, window_size=100)
    assert report.sample_size == 0
    assert report.overall_severity == "green"
    assert report.feature_drifts == []


@pytest.mark.asyncio
async def test_recompute_drift_assembles_window_for_matching_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle_with(stats=_ref_stats())

    async def fake_fetch(_session, _limit):
        return [_row(version="1", t_offset=i, score=0.4 + (i % 3) * 0.05) for i in range(50)]

    monkeypatch.setattr("app.services.drift_service.fetch_recent", fake_fetch)

    report = await recompute_drift(AsyncMock(), bundle, window_size=100)
    assert report.sample_size == 50
    # window edges sourced from row timestamps
    assert report.window_start <= report.window_end
    # numeric + categorical drifts are present
    assert any(fd.feature_type == "numeric" for fd in report.feature_drifts)
    assert any(fd.feature_type == "categorical" for fd in report.feature_drifts)
