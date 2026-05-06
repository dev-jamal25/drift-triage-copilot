"""Route + lifespan tests for the /predict endpoint.

Two layers:
 * Route-level: build a ModelBundle in-process (no MLflow) and override the
   ``get_model_bundle`` dependency so the test runs in milliseconds.
 * Integration: register a tiny model into a tmp ``file://`` MLflow URI, then
   call ``load_bundle`` directly to confirm the lifespan path works against
   a real registry.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.core.config import Settings
from app.dependencies import get_model_bundle
from app.main import app
from app.services.model_loader import ModelBundle, load_bundle
from ml.data import clean
from ml.evaluate import evaluate_at_threshold
from ml.features import build_preprocessor
from ml.registry import log_and_register


def _synthetic_raw(n: int = 600, seed: int = 0) -> pd.DataFrame:
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


def _fitted_pipe(cleaned: pd.DataFrame) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=200, solver="lbfgs")),
        ]
    )
    pipe.fit(cleaned.drop(columns=["target"]), cleaned["target"])
    return pipe


def _canonical_payload() -> dict[str, object]:
    return {
        "age": 35,
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


@pytest.fixture
def fitted_bundle() -> ModelBundle:
    cleaned = clean(_synthetic_raw(seed=11))
    pipe = _fitted_pipe(cleaned)
    return ModelBundle(
        pipeline=pipe,
        threshold=0.5,
        model_uri="models:/test-fixture@staging",
        model_name="test-fixture",
        version="42",
        alias="staging",
        run_id="fixture-run",
        loaded_at=datetime.now(UTC),
    )


@pytest.fixture
def client(fitted_bundle: ModelBundle) -> Iterator[TestClient]:
    """TestClient with the model bundle dependency overridden — no lifespan needed."""
    app.dependency_overrides[get_model_bundle] = lambda: fitted_bundle
    try:
        # No `with` -> lifespan does not run, so MLflow is never contacted.
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_predict_happy_path(client: TestClient) -> None:
    resp = client.post("/predict", json=_canonical_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["label"] in (0, 1)
    assert body["threshold"] == 0.5
    assert body["model_name"] == "test-fixture"
    assert body["model_version"] == "42"
    # label respects threshold
    if body["score"] >= 0.5:
        assert body["label"] == 1
    else:
        assert body["label"] == 0


def test_predict_rejects_extra_field(client: TestClient) -> None:
    payload = _canonical_payload()
    payload["duration"] = 100
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_missing_field(client: TestClient) -> None:
    payload = _canonical_payload()
    del payload["age"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_503_when_bundle_missing() -> None:
    """If lifespan didn't load the bundle, /predict returns 503 — not 500."""
    app.dependency_overrides.clear()
    raw = TestClient(app)
    resp = raw.post("/predict", json=_canonical_payload())
    assert resp.status_code == 503


def test_healthz_works_without_lifespan() -> None:
    raw = TestClient(app)
    resp = raw.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readyz_reports_not_ready_without_lifespan() -> None:
    # Clear any state from prior test runs that may have set app.state.model_bundle
    if hasattr(app.state, "model_bundle"):
        delattr(app.state, "model_bundle")
    raw = TestClient(app)
    resp = raw.get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "not_ready"


def test_load_bundle_round_trip_via_real_registry(tmp_path: Path) -> None:
    """Register a tiny model in a tmp MLflow URI, then load_bundle it back."""
    cleaned = clean(_synthetic_raw(seed=22))
    pipe = _fitted_pipe(cleaned)
    X = cleaned.drop(columns=["target"])
    y = cleaned["target"]
    proba = pipe.predict_proba(X)[:, 1]
    metrics = evaluate_at_threshold(y.to_numpy(), proba, threshold=0.5)

    tracking_uri = (tmp_path / "mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri)

    log_and_register(
        pipe,
        model_name="bundle-round-trip",
        model_family="logreg",
        threshold=0.5,
        min_recall=0.5,
        cleaned_df_for_hash=cleaned,
        sample_X=X,
        val_metrics=metrics,
        test_metrics=metrics,
        n_train=int(0.6 * len(cleaned)),
        n_val=int(0.2 * len(cleaned)),
        n_test=int(0.2 * len(cleaned)),
        positive_rate=float(y.mean()),
        tracking_uri=tracking_uri,
        experiment_name="lifespan-load-test",
        stage="Staging",
    )

    settings = Settings(
        mlflow_tracking_uri=tracking_uri,
        model_name="bundle-round-trip",
        model_alias="staging",
        load_model_on_startup=True,
    )
    bundle = load_bundle(settings)

    assert bundle.threshold == pytest.approx(0.5)
    assert bundle.version == "1"
    assert bundle.alias == "staging"
    assert bundle.model_uri == "models:/bundle-round-trip@staging"

    # Loaded pipeline produces predictions in the right shape
    out = bundle.pipeline.predict_proba(X.head(3))
    assert out.shape == (3, 2)
