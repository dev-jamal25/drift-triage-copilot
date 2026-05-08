"""Tests for ``ml.drift_stats.compute_reference_stats``.

Pure unit — no MLflow, no DB. Verifies the schema of the returned dict and
the invariants that Step 3's drift-metrics layer will rely on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, clean
from ml.drift_stats import compute_reference_stats


def _synthetic_raw(n: int = 800, seed: int = 0) -> pd.DataFrame:
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


def _ref_stats_from_synthetic(seed: int = 1, n: int = 800):
    cleaned = clean(_synthetic_raw(n=n, seed=seed))
    X = cleaned.drop(columns=["target"])
    y = cleaned["target"]
    proba = np.clip(np.linspace(0.0, 1.0, num=len(X)), 0.0, 1.0)
    return compute_reference_stats(X, proba, positive_rate=float(y.mean())), X, y


def test_reference_stats_schema_keys() -> None:
    stats, _, _ = _ref_stats_from_synthetic()
    for key in (
        "n_samples",
        "positive_rate",
        "computed_at",
        "n_bins",
        "numeric",
        "categorical",
        "output",
    ):
        assert key in stats


def test_n_samples_matches_input_length() -> None:
    stats, X, _ = _ref_stats_from_synthetic()
    assert stats["n_samples"] == len(X)


def test_positive_rate_recorded() -> None:
    stats, _, y = _ref_stats_from_synthetic()
    assert stats["positive_rate"] == float(y.mean())


def test_numeric_features_all_covered() -> None:
    stats, _, _ = _ref_stats_from_synthetic()
    for col in NUMERIC_COLUMNS:
        assert col in stats["numeric"], f"missing numeric ref stats for {col}"
        bucket = stats["numeric"][col]
        assert "bin_edges" in bucket and "ref_counts" in bucket
        assert len(bucket["bin_edges"]) == len(bucket["ref_counts"]) + 1
        assert all(isinstance(c, int) for c in bucket["ref_counts"])


def test_numeric_counts_sum_to_n_samples() -> None:
    stats, _, _ = _ref_stats_from_synthetic()
    n = stats["n_samples"]
    for col, bucket in stats["numeric"].items():
        assert sum(bucket["ref_counts"]) == n, f"{col} counts don't sum"


def test_categorical_features_all_covered() -> None:
    stats, _, _ = _ref_stats_from_synthetic()
    for col in CATEGORICAL_COLUMNS:
        assert col in stats["categorical"]
        ref = stats["categorical"][col]["ref_counts"]
        assert isinstance(ref, dict)
        assert all(isinstance(k, str) and isinstance(v, int) for k, v in ref.items())


def test_categorical_counts_sum_to_n_samples() -> None:
    stats, _, _ = _ref_stats_from_synthetic()
    n = stats["n_samples"]
    for bucket in stats["categorical"].values():
        assert sum(bucket["ref_counts"].values()) == n


def test_output_bins_present_and_summed() -> None:
    stats, _, _ = _ref_stats_from_synthetic()
    out = stats["output"]
    assert "bin_edges" in out and "ref_counts" in out
    assert len(out["bin_edges"]) == len(out["ref_counts"]) + 1
    assert sum(out["ref_counts"]) == stats["n_samples"]


def test_unknown_level_preserved_in_categorical_stats() -> None:
    raw = _synthetic_raw(n=400, seed=7)
    raw.loc[0, "job"] = "unknown"
    cleaned = clean(raw)
    X = cleaned.drop(columns=["target"])
    proba = np.full(len(X), 0.5)
    stats = compute_reference_stats(X, proba, positive_rate=0.2)
    assert "unknown" in stats["categorical"]["job"]["ref_counts"]


def test_constant_numeric_column_does_not_explode() -> None:
    """All-equal columns produce a single one-wide bucket — no division-by-zero."""
    raw = _synthetic_raw(n=300, seed=3)
    raw["age"] = 42  # collapse a numeric column to a constant
    cleaned = clean(raw)
    X = cleaned.drop(columns=["target"])
    proba = np.full(len(X), 0.3)
    stats = compute_reference_stats(X, proba, positive_rate=0.2)
    age_bucket = stats["numeric"]["age"]
    assert sum(age_bucket["ref_counts"]) == len(X)
    assert len(age_bucket["bin_edges"]) >= 2
