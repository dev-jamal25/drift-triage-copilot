"""Tests for MLflow logging, registration, load-back, and fidelity check.

Each test sets a per-test ``file://`` tracking URI inside ``tmp_path`` so the
suite never touches a shared mlruns/ directory or hits the network.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.data import clean
from ml.evaluate import evaluate_at_threshold
from ml.features import build_preprocessor
from ml.registry import (
    build_model_schema,
    compute_dataset_hash,
    compute_env_fingerprint,
    fidelity_check,
    load_model,
    log_and_register,
    render_model_card,
)


def _synthetic_raw(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Small UCI-shaped raw frame; cleaned form is the test corpus."""
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


def _fitted_pipe(df: pd.DataFrame) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=200, solver="lbfgs")),
        ]
    )
    X = df.drop(columns=["target"])
    y = df["target"]
    pipe.fit(X, y)
    return pipe


@pytest.fixture(autouse=True)
def _isolate_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Force every test to use its own file:// tracking URI."""
    uri = (tmp_path / "mlruns").as_uri()
    mlflow.set_tracking_uri(uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    yield uri


def test_compute_dataset_hash_is_deterministic() -> None:
    df = clean(_synthetic_raw(200, seed=7))
    assert compute_dataset_hash(df) == compute_dataset_hash(df.copy())


def test_compute_dataset_hash_differs_when_data_changes() -> None:
    a = clean(_synthetic_raw(200, seed=7))
    b = clean(_synthetic_raw(200, seed=8))
    assert compute_dataset_hash(a) != compute_dataset_hash(b)


def test_compute_env_fingerprint_has_required_keys() -> None:
    fp = compute_env_fingerprint()
    for key in ("python", "platform", "sklearn", "mlflow", "pandas", "numpy"):
        assert key in fp and fp[key]


def test_build_model_schema_describes_inputs_and_outputs() -> None:
    cleaned = clean(_synthetic_raw(300, seed=1))
    pipe = _fitted_pipe(cleaned)
    schema = build_model_schema(pipe)

    assert "input" in schema and "output" in schema
    assert "numeric" in schema["input"] and "categorical" in schema["input"]
    assert "age" in schema["input"]["numeric"]
    assert "job" in schema["input"]["categorical"]
    # OneHot levels reflect the data the encoder actually saw
    assert "unknown" in schema["input"]["categorical"]["job"]
    assert schema["class_labels"] == {"0": "no", "1": "yes"}


def test_render_model_card_substitutes_fields(tmp_path: Path) -> None:
    template_path = Path(__file__).resolve().parents[1] / "ml" / "model_card.md"
    val_metrics = {
        "accuracy": 0.7,
        "f1_macro": 0.6,
        "roc_auc": 0.8,
        "precision": 0.3,
        "recall": 0.75,
    }
    rendered = render_model_card(
        template_path,
        model_name="test-model",
        model_version="1",
        model_family="logreg",
        threshold=0.42,
        min_recall=0.75,
        alias="staging",
        n_train=10,
        n_val=5,
        n_test=5,
        positive_rate=0.2,
        dataset_hash="abc123",
        val_metrics=val_metrics,
        test_metrics=val_metrics,
        env_fingerprint={"python": "3.12.0", "sklearn": "1.5.0"},
    )
    assert "test-model" in rendered
    assert "abc123" in rendered
    assert "0.4200" in rendered  # threshold formatted
    assert "logreg" in rendered


def test_log_and_register_round_trip_with_fidelity(tmp_path: Path) -> None:
    cleaned = clean(_synthetic_raw(500, seed=2))
    pipe = _fitted_pipe(cleaned)
    X = cleaned.drop(columns=["target"])
    y = cleaned["target"]

    proba = pipe.predict_proba(X)[:, 1]
    val_metrics = evaluate_at_threshold(y.to_numpy(), proba, threshold=0.5)
    test_metrics = evaluate_at_threshold(y.to_numpy(), proba, threshold=0.5)

    run = log_and_register(
        pipe,
        model_name="test-classifier",
        model_family="logreg",
        threshold=0.5,
        min_recall=0.5,
        cleaned_df_for_hash=cleaned,
        sample_X=X,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        n_train=300,
        n_val=100,
        n_test=100,
        positive_rate=float(y.mean()),
        tracking_uri=mlflow.get_tracking_uri(),
        experiment_name="unit-test-exp",
        stage="Staging",
    )

    assert run.alias == "staging"
    assert run.version == "1"
    assert run.model_uri == "models:/test-classifier@staging"

    loaded = load_model(run.model_uri)
    summary = fidelity_check(pipe, loaded, X.head(50))
    assert summary["max_abs_diff"] <= summary["atol"]


def test_log_and_register_rejects_unknown_stage() -> None:
    cleaned = clean(_synthetic_raw(200, seed=3))
    pipe = _fitted_pipe(cleaned)
    X = cleaned.drop(columns=["target"])
    y = cleaned["target"]
    proba = pipe.predict_proba(X)[:, 1]
    metrics = evaluate_at_threshold(y.to_numpy(), proba, 0.5)

    with pytest.raises(ValueError, match="Unknown stage"):
        log_and_register(
            pipe,
            model_name="test-bad-stage",
            model_family="logreg",
            threshold=0.5,
            min_recall=0.5,
            cleaned_df_for_hash=cleaned,
            sample_X=X,
            val_metrics=metrics,
            test_metrics=metrics,
            n_train=100,
            n_val=50,
            n_test=50,
            positive_rate=float(y.mean()),
            tracking_uri=mlflow.get_tracking_uri(),
            experiment_name="unit-test-bad-stage",
            stage="Live",
        )


def test_fidelity_check_raises_on_diverging_models() -> None:
    cleaned = clean(_synthetic_raw(300, seed=4))
    pipe_a = _fitted_pipe(cleaned)
    # Different classifier => predict_proba will differ
    pipe_b = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=200, C=0.001)),
        ]
    )
    pipe_b.fit(cleaned.drop(columns=["target"]), cleaned["target"])

    with pytest.raises(AssertionError, match="Fidelity check failed"):
        fidelity_check(pipe_a, pipe_b, cleaned.drop(columns=["target"]).head(50))


def test_load_model_rejects_plain_name() -> None:
    with pytest.raises(ValueError, match="models:"):
        load_model("just-a-name")
