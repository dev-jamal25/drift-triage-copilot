"""Tests for ``ml.drift_metrics``: PSI, chi², severity buckets, and the
``compute_drift_report`` builder.

Pure unit. No DB, no MLflow, no API.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ml.data import clean
from ml.drift_metrics import (
    bucket_chi2,
    bucket_psi,
    categorical_feature_drift,
    chi2_pvalue,
    compute_drift_report,
    max_severity,
    numeric_feature_drift,
    output_drift_psi,
    psi,
)
from ml.drift_stats import compute_reference_stats

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


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


def _ref_and_live(seed_ref: int = 1, seed_live: int = 1, n_ref: int = 1500, n_live: int = 800):
    """Build (reference_stats, live_features, live_proba) for end-to-end tests."""
    cleaned_ref = clean(_synthetic_raw(n=n_ref, seed=seed_ref))
    X_ref = cleaned_ref.drop(columns=["target"])
    y_ref = cleaned_ref["target"]
    rng = np.random.default_rng(seed_ref + 100)
    proba_ref = rng.random(len(X_ref))
    ref_stats = compute_reference_stats(X_ref, proba_ref, positive_rate=float(y_ref.mean()))

    cleaned_live = clean(_synthetic_raw(n=n_live, seed=seed_live))
    X_live = cleaned_live.drop(columns=["target"])
    rng = np.random.default_rng(seed_live + 200)
    proba_live = rng.random(len(X_live))
    return ref_stats, X_live, proba_live


# ----------------------------------------------------------------------------
# psi
# ----------------------------------------------------------------------------


def test_psi_zero_for_identical_distributions() -> None:
    ref = np.array([0.1, 0.2, 0.3, 0.4])
    assert psi(ref, ref) == pytest.approx(0.0, abs=1e-12)


def test_psi_increases_with_shift() -> None:
    ref = np.array([0.25, 0.25, 0.25, 0.25])
    mild = np.array([0.20, 0.30, 0.25, 0.25])
    severe = np.array([0.05, 0.45, 0.05, 0.45])
    assert psi(ref, mild) < psi(ref, severe)
    assert psi(ref, severe) > 0.25  # firmly in the "red" zone


def test_psi_handles_zero_bins_without_explosion() -> None:
    """An empty bin in either side should not raise."""
    ref = np.array([0.5, 0.5, 0.0])
    live = np.array([0.0, 0.5, 0.5])
    val = psi(ref, live)
    assert np.isfinite(val)
    assert val > 0


def test_psi_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        psi([0.5, 0.5], [0.25, 0.25, 0.5])


# ----------------------------------------------------------------------------
# chi2_pvalue
# ----------------------------------------------------------------------------


def test_chi2_pvalue_high_for_proportional_distributions() -> None:
    ref = [100, 200, 300]
    live = [10, 20, 30]  # same shape, scaled down
    assert chi2_pvalue(ref, live) > 0.99


def test_chi2_pvalue_low_for_clearly_shifted_distribution() -> None:
    ref = [500, 500, 0]
    live = [10, 10, 480]  # mass moved to a level that was empty in ref
    assert chi2_pvalue(ref, live) < 0.001


def test_chi2_pvalue_returns_one_for_empty_live_window() -> None:
    assert chi2_pvalue([10, 20, 30], [0, 0, 0]) == 1.0


# ----------------------------------------------------------------------------
# bucket_psi / bucket_chi2 / max_severity
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, "green"),
        (0.099, "green"),
        (0.1, "yellow"),
        (0.25, "yellow"),
        (0.2501, "red"),
        (1.0, "red"),
    ],
)
def test_bucket_psi_thresholds(value: float, expected: str) -> None:
    assert bucket_psi(value) == expected


@pytest.mark.parametrize(
    "p,expected",
    [
        (0.5, "green"),
        (0.0101, "green"),
        (0.01, "green"),  # boundary is strict (< 0.01) per DECISIONS.md
        (0.0099, "yellow"),
        (0.001, "yellow"),  # boundary is strict (< 0.001) per DECISIONS.md
        (0.0009, "red"),
        (0.0, "red"),
    ],
)
def test_bucket_chi2_thresholds(p: float, expected: str) -> None:
    assert bucket_chi2(p) == expected


def test_max_severity_picks_worst() -> None:
    assert max_severity(["green", "yellow", "red"]) == "red"
    assert max_severity(["green", "yellow"]) == "yellow"
    assert max_severity(["green"]) == "green"
    assert max_severity([]) == "green"


# ----------------------------------------------------------------------------
# Per-feature drift helpers
# ----------------------------------------------------------------------------


def test_numeric_feature_drift_green_on_same_distribution() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, size=2000)
    ref_stats = compute_reference_stats(
        pd.DataFrame({"age": values}),
        np.zeros(len(values)),
        positive_rate=0.0,
    )
    fd = numeric_feature_drift("age", ref_stats["numeric"]["age"], values)
    assert fd.feature_type == "numeric"
    assert fd.metric == "psi"
    assert fd.severity == "green"
    assert fd.value < 0.05


def test_numeric_feature_drift_red_on_severe_shift() -> None:
    rng = np.random.default_rng(0)
    ref_values = rng.normal(0, 1, size=2000)
    live_values = rng.normal(5, 1, size=2000)  # mean shifted by 5 std
    ref_stats = compute_reference_stats(
        pd.DataFrame({"age": ref_values}),
        np.zeros(len(ref_values)),
        positive_rate=0.0,
    )
    fd = numeric_feature_drift("age", ref_stats["numeric"]["age"], live_values)
    assert fd.severity == "red"


def test_categorical_feature_drift_green_on_same_distribution() -> None:
    rng = np.random.default_rng(1)
    levels = rng.choice(["a", "b", "c"], size=2000, p=[0.5, 0.3, 0.2])
    ref_stats = compute_reference_stats(
        pd.DataFrame({"job": levels}).assign(  # need numeric col for ref_stats
            age=np.zeros(2000)
        ),
        np.zeros(2000),
        positive_rate=0.0,
    )
    live = pd.Series(rng.choice(["a", "b", "c"], size=2000, p=[0.5, 0.3, 0.2]))
    fd = categorical_feature_drift("job", ref_stats["categorical"]["job"], live)
    assert fd.feature_type == "categorical"
    assert fd.metric == "chi2"
    assert fd.severity == "green"


def test_categorical_feature_drift_red_on_severe_shift() -> None:
    rng = np.random.default_rng(2)
    ref_levels = rng.choice(["a", "b", "c"], size=2000, p=[0.5, 0.3, 0.2])
    ref_stats = compute_reference_stats(
        pd.DataFrame({"job": ref_levels, "age": np.zeros(2000)}),
        np.zeros(2000),
        positive_rate=0.0,
    )
    live = pd.Series(rng.choice(["a", "b", "c"], size=2000, p=[0.05, 0.05, 0.9]))
    fd = categorical_feature_drift("job", ref_stats["categorical"]["job"], live)
    assert fd.severity == "red"


def test_output_drift_psi_zero_when_proba_unchanged() -> None:
    rng = np.random.default_rng(0)
    proba = rng.uniform(0, 1, size=2000)
    ref_stats = compute_reference_stats(
        pd.DataFrame({"age": np.zeros(len(proba))}),
        proba,
        positive_rate=0.0,
    )
    val = output_drift_psi(ref_stats["output"], proba)
    assert val < 0.05


def test_output_drift_psi_high_when_scores_collapse_to_one_bin() -> None:
    rng = np.random.default_rng(0)
    ref_proba = rng.uniform(0, 1, size=2000)
    live_proba = np.full(2000, 0.95)
    ref_stats = compute_reference_stats(
        pd.DataFrame({"age": np.zeros(len(ref_proba))}),
        ref_proba,
        positive_rate=0.0,
    )
    val = output_drift_psi(ref_stats["output"], live_proba)
    assert val > 0.25


# ----------------------------------------------------------------------------
# compute_drift_report
# ----------------------------------------------------------------------------


def test_compute_drift_report_empty_window_returns_green() -> None:
    ref_stats, _, _ = _ref_and_live()
    now = datetime.now(UTC)
    report = compute_drift_report(
        ref_stats,
        recent_features=pd.DataFrame(),
        recent_proba_positive=np.array([]),
        window_start=now - timedelta(minutes=1),
        window_end=now,
    )
    assert report.sample_size == 0
    assert report.overall_severity == "green"
    assert report.feature_drifts == []
    assert report.output_drift_psi == 0.0


def test_compute_drift_report_same_distribution_reports_green_overall() -> None:
    ref_stats, X_live, proba_live = _ref_and_live(seed_ref=11, seed_live=11)
    now = datetime.now(UTC)
    report = compute_drift_report(
        ref_stats,
        recent_features=X_live,
        recent_proba_positive=proba_live,
        window_start=now - timedelta(minutes=10),
        window_end=now,
    )
    assert report.sample_size == len(X_live)
    # Numeric and categorical drifts should all be present in the report.
    assert any(fd.feature_type == "numeric" for fd in report.feature_drifts)
    assert any(fd.feature_type == "categorical" for fd in report.feature_drifts)
    assert report.overall_severity == "green"


def test_compute_drift_report_flips_red_on_engineered_shift() -> None:
    ref_stats, X_live, proba_live = _ref_and_live(seed_ref=11, seed_live=11)
    # Shift one numeric column hard — 100 z-scores away from the reference.
    X_shifted = X_live.copy()
    X_shifted["age"] = X_shifted["age"] + 1000
    # And collapse the output distribution to a single high bucket.
    proba_shifted = np.full(len(X_shifted), 0.97)

    now = datetime.now(UTC)
    report = compute_drift_report(
        ref_stats,
        recent_features=X_shifted,
        recent_proba_positive=proba_shifted,
        window_start=now - timedelta(minutes=10),
        window_end=now,
    )
    assert report.overall_severity == "red"
    age_fd = next(fd for fd in report.feature_drifts if fd.feature_name == "age")
    assert age_fd.severity == "red"
    assert report.output_drift_psi > 0.25


def test_compute_drift_report_unseen_categorical_level_is_handled() -> None:
    """A new categorical level in the live window must not crash chi²."""
    ref_stats, X_live, proba_live = _ref_and_live(seed_ref=11, seed_live=11)
    X_live = X_live.copy()
    # introduce a brand-new level in `job` for the entire live window
    X_live["job"] = "brand-new-level"
    now = datetime.now(UTC)
    report = compute_drift_report(
        ref_stats,
        recent_features=X_live,
        recent_proba_positive=proba_live,
        window_start=now - timedelta(minutes=1),
        window_end=now,
    )
    job_fd = next(fd for fd in report.feature_drifts if fd.feature_name == "job")
    assert job_fd.feature_type == "categorical"
    # New-level-only live window is a strong signal — should be red.
    assert job_fd.severity == "red"
