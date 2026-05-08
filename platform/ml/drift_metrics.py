"""Drift metrics over a reference distribution and a live window.

Pure functions only — no DB, no MLflow, no API. ``compute_drift_report``
takes the reference stats produced by ``ml.drift_stats.compute_reference_stats``
plus a live window of features + scores and returns a
``shared.contracts.DriftReport``.

Severity bins per ``DECISIONS.md``:

  * PSI:    < 0.10 green | 0.10–0.25 yellow | > 0.25 red
  * Chi²:   p < 0.001 red | p < 0.01 yellow | else green
  * Output: same PSI bins as numeric features
  * Overall = max severity across all features and output drift
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from shared.contracts import DriftReport, FeatureDrift, Severity

# Floor for PSI's frequency arrays so log/division don't blow up on empty bins.
_FREQ_EPSILON = 1e-6
# Floor for chi² expected counts; avoids "expected frequency must be > 0" errors
# when the live window has new categorical levels not seen in training.
_CHI2_EXPECTED_FLOOR = 0.5

_SEVERITY_RANK: dict[Severity, int] = {"green": 0, "yellow": 1, "red": 2}


# =============================================================================
# Primitives
# =============================================================================


def psi(ref_freq: np.ndarray | list[float], live_freq: np.ndarray | list[float]) -> float:
    """Population Stability Index between two same-shape frequency arrays.

    Both inputs are clamped to ``_FREQ_EPSILON`` to avoid log(0) / div-by-zero.
    Inputs do not need to sum to exactly 1 — the formula is symmetric in scale,
    but typically you pass normalized frequencies for interpretability.
    """
    ref = np.asarray(ref_freq, dtype=float)
    live = np.asarray(live_freq, dtype=float)
    if ref.shape != live.shape:
        raise ValueError(f"psi: shape mismatch ref={ref.shape} live={live.shape}")
    if ref.size == 0:
        return 0.0
    ref = np.clip(ref, _FREQ_EPSILON, None)
    live = np.clip(live, _FREQ_EPSILON, None)
    return float(np.sum((live - ref) * np.log(live / ref)))


def chi2_pvalue(
    ref_counts: np.ndarray | list[float],
    live_counts: np.ndarray | list[float],
) -> float:
    """Chi² test of two count vectors, returning the p-value.

    Reference counts are rescaled to the live window's total so the test
    asks "is the live distribution consistent with the reference shape?"
    Empty live windows return ``1.0`` (cannot reject H0).
    """
    ref = np.asarray(ref_counts, dtype=float)
    live = np.asarray(live_counts, dtype=float)
    if ref.shape != live.shape:
        raise ValueError(f"chi2_pvalue: shape mismatch ref={ref.shape} live={live.shape}")
    if live.sum() <= 0 or ref.sum() <= 0:
        return 1.0

    expected = ref * (live.sum() / ref.sum())
    expected = np.clip(expected, _CHI2_EXPECTED_FLOOR, None)
    # Re-normalise expected to live total so chisquare's sum-equality check passes.
    expected *= live.sum() / expected.sum()
    _, p_value = stats.chisquare(f_obs=live, f_exp=expected)
    return float(p_value)


def bucket_psi(value: float) -> Severity:
    """Map a PSI value to a severity bin (DECISIONS.md)."""
    if value < 0.1:
        return "green"
    if value <= 0.25:
        return "yellow"
    return "red"


def bucket_chi2(p_value: float) -> Severity:
    """Map a chi² p-value to a severity bin (DECISIONS.md)."""
    if p_value < 0.001:
        return "red"
    if p_value < 0.01:
        return "yellow"
    return "green"


def max_severity(severities: list[Severity]) -> Severity:
    """Reduce a list of severities to the worst observed."""
    if not severities:
        return "green"
    return max(severities, key=_SEVERITY_RANK.__getitem__)


# =============================================================================
# Per-feature drift
# =============================================================================


def numeric_feature_drift(
    feature_name: str,
    ref_bucket: dict[str, Any],
    live_values: np.ndarray | pd.Series,
) -> FeatureDrift:
    """PSI on a numeric column, using the reference bin edges."""
    edges = np.asarray(ref_bucket["bin_edges"], dtype=float)
    ref_counts = np.asarray(ref_bucket["ref_counts"], dtype=float)
    arr = np.asarray(live_values, dtype=float)
    arr = arr[~np.isnan(arr)]

    live_counts, _ = np.histogram(arr, bins=edges)
    ref_total = float(ref_counts.sum()) or 1.0
    live_total = float(live_counts.sum()) or 1.0
    psi_val = psi(ref_counts / ref_total, live_counts / live_total)
    return FeatureDrift(
        feature_name=feature_name,
        feature_type="numeric",
        metric="psi",
        value=psi_val,
        severity=bucket_psi(psi_val),
    )


def categorical_feature_drift(
    feature_name: str,
    ref_bucket: dict[str, Any],
    live_values: pd.Series,
) -> FeatureDrift:
    """Chi² on a categorical column.

    The level set is the union of reference levels and live levels — new
    levels in the live window get count 0 in the reference (handled by the
    expected-count floor), and missing reference levels get count 0 in live.
    """
    ref_counts: dict[str, int] = ref_bucket["ref_counts"]
    live_counts_raw = live_values.astype(str).value_counts().to_dict()

    keys = sorted(set(ref_counts) | set(live_counts_raw))
    ref_arr = np.array([ref_counts.get(k, 0) for k in keys], dtype=float)
    live_arr = np.array([live_counts_raw.get(k, 0) for k in keys], dtype=float)

    p_value = chi2_pvalue(ref_arr, live_arr)
    return FeatureDrift(
        feature_name=feature_name,
        feature_type="categorical",
        metric="chi2",
        value=p_value,
        severity=bucket_chi2(p_value),
    )


def output_drift_psi(
    output_section: dict[str, Any],
    live_proba_positive: np.ndarray,
) -> float:
    """PSI between the reference output distribution and the live one."""
    edges = np.asarray(output_section["bin_edges"], dtype=float)
    ref_counts = np.asarray(output_section["ref_counts"], dtype=float)
    live_counts, _ = np.histogram(np.asarray(live_proba_positive, dtype=float), bins=edges)
    ref_total = float(ref_counts.sum()) or 1.0
    live_total = float(live_counts.sum()) or 1.0
    return psi(ref_counts / ref_total, live_counts / live_total)


# =============================================================================
# Report builder
# =============================================================================


def compute_drift_report(
    reference_stats: dict[str, Any],
    recent_features: pd.DataFrame,
    recent_proba_positive: np.ndarray,
    *,
    window_start: datetime,
    window_end: datetime,
) -> DriftReport:
    """Build a ``DriftReport`` over the live window vs. the reference baseline.

    ``recent_features`` is expected to be already in *cleaned* form — i.e.
    after ``apply_pdays_sentinel``, with no ``duration`` column. It's the
    caller's job (Step 4 — drift service) to assemble that frame from the
    ``predictions_log.features`` JSON column.
    """
    n_samples = int(len(recent_features))

    if n_samples == 0:
        return DriftReport(
            window_start=window_start,
            window_end=window_end,
            sample_size=0,
            overall_severity="green",
            feature_drifts=[],
            output_drift_psi=0.0,
        )

    feature_drifts: list[FeatureDrift] = []

    for col, bucket in reference_stats.get("numeric", {}).items():
        if col not in recent_features.columns:
            continue
        feature_drifts.append(numeric_feature_drift(col, bucket, recent_features[col].to_numpy()))

    for col, bucket in reference_stats.get("categorical", {}).items():
        if col not in recent_features.columns:
            continue
        feature_drifts.append(categorical_feature_drift(col, bucket, recent_features[col]))

    output_section = reference_stats.get("output") or {}
    out_psi = (
        output_drift_psi(output_section, recent_proba_positive)
        if output_section and len(recent_proba_positive)
        else 0.0
    )

    severities: list[Severity] = [fd.severity for fd in feature_drifts]
    severities.append(bucket_psi(out_psi))

    return DriftReport(
        window_start=window_start,
        window_end=window_end,
        sample_size=n_samples,
        overall_severity=max_severity(severities),
        feature_drifts=feature_drifts,
        output_drift_psi=out_psi,
    )
