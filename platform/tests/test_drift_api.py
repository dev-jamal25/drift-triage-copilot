"""Route test for ``GET /drift``.

Overrides the model-bundle and drift-scheduler dependencies so the route runs
without lifespan, MLflow, or a real DB.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from shared.contracts import DriftReport, FeatureDrift

from app.dependencies import get_drift_scheduler, get_model_bundle
from app.main import app
from app.services.drift_scheduler import DriftScheduler
from app.services.model_loader import ModelBundle


def _bundle() -> ModelBundle:
    return ModelBundle(
        pipeline=MagicMock(),
        threshold=0.5,
        model_uri="models:/test@staging",
        model_name="test",
        version="42",
        alias="staging",
        run_id="r-test",
        loaded_at=datetime.now(UTC),
        reference_stats={"numeric": {}, "categorical": {}, "output": {}},
    )


def _empty_scheduler() -> DriftScheduler:
    sch = DriftScheduler.__new__(DriftScheduler)
    sch.latest_report = None
    sch.previous_severity = None
    sch.last_refreshed_at = None
    return sch


def _scheduler_with_report(severity: str = "yellow") -> DriftScheduler:
    sch = DriftScheduler.__new__(DriftScheduler)
    now = datetime.now(UTC)
    sch.latest_report = DriftReport(
        window_start=now,
        window_end=now,
        sample_size=42,
        overall_severity=severity,  # type: ignore[arg-type]
        feature_drifts=[
            FeatureDrift(
                feature_name="age",
                feature_type="numeric",
                metric="psi",
                value=0.18,
                severity="yellow",
            )
        ],
        output_drift_psi=0.05,
    )
    sch.previous_severity = "green"
    sch.last_refreshed_at = now
    return sch


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_model_bundle] = _bundle
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_drift_returns_503_before_first_tick(client: TestClient) -> None:
    app.dependency_overrides[get_drift_scheduler] = _empty_scheduler
    resp = client.get("/drift")
    assert resp.status_code == 503


def test_drift_returns_200_with_cached_report(client: TestClient) -> None:
    app.dependency_overrides[get_drift_scheduler] = lambda: _scheduler_with_report("yellow")
    resp = client.get("/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_name"] == "test"
    assert body["model_version"] == "42"
    assert body["drift"]["overall_severity"] == "yellow"
    assert body["drift"]["sample_size"] == 42
    assert body["drift"]["output_drift_psi"] == 0.05
    feats = body["drift"]["feature_drifts"]
    assert len(feats) == 1
    assert feats[0]["feature_name"] == "age"
    assert feats[0]["metric"] == "psi"


def test_drift_503_when_scheduler_missing_from_state() -> None:
    """No dependency override + no lifespan → ``get_drift_scheduler`` returns 503."""
    app.dependency_overrides.clear()
    raw = TestClient(app)
    resp = raw.get("/drift")
    assert resp.status_code == 503
