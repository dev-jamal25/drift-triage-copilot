"""Reference statistics for drift detection.

Computed once on the training fold during ``python -m ml.train`` and logged
to MLflow as ``reference_stats.json`` next to the model. The drift scheduler
(future step) loads these from the model's MLflow run and compares live
windows against them: PSI on numeric + output bins, chi² on categorical
counts.

Schema (also documented inline in ``compute_reference_stats``):

    {
      "n_samples": int,
      "positive_rate": float,
      "computed_at": iso8601 utc,
      "numeric": {
        "<col>": {"bin_edges": [float, ...], "ref_counts": [int, ...]},
        ...
      },
      "categorical": {
        "<col>": {"ref_counts": {"<level>": int, ...}},
        ...
      },
      "output": {"bin_edges": [float, ...], "ref_counts": [int, ...]}
    }

The bin counts sum to ``n_samples`` per feature (modulo a small number of
NaNs which are dropped — there shouldn't be any post-clean).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ml.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

DEFAULT_N_BINS = 10


def _numeric_bins(
    values: np.ndarray, n_bins: int = DEFAULT_N_BINS
) -> tuple[list[float], list[int]]:
    """Quantile-bin a 1-D array. Duplicate edges are collapsed."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return [0.0, 1.0], [0]

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(arr, quantiles))
    if edges.size < 2:
        # Constant column — give it a 1-wide bucket so histogram still works.
        edges = np.array([float(edges[0]), float(edges[0]) + 1.0])

    counts, _ = np.histogram(arr, bins=edges)
    return [float(e) for e in edges], [int(c) for c in counts]


def _categorical_counts(values: pd.Series) -> dict[str, int]:
    """Return ``{level: count}`` over the column's observed levels."""
    return {str(level): int(count) for level, count in values.value_counts().items()}


def compute_reference_stats(
    X_train: pd.DataFrame,
    train_proba_positive: np.ndarray,
    *,
    positive_rate: float,
    n_bins: int = DEFAULT_N_BINS,
) -> dict[str, Any]:
    """Build the reference-stats payload from the train fold.

    Args:
        X_train: cleaned training features (post-``apply_pdays_sentinel``,
            without the target column).
        train_proba_positive: ``predict_proba(X_train)[:, 1]`` from the fitted
            pipeline.
        positive_rate: train-fold positive rate, recorded for visibility.
        n_bins: bins per numeric/output feature (default 10).
    """
    numeric: dict[str, dict[str, list[float] | list[int]]] = {}
    for col in NUMERIC_COLUMNS:
        if col not in X_train.columns:
            continue
        edges, counts = _numeric_bins(X_train[col].to_numpy(), n_bins=n_bins)
        numeric[col] = {"bin_edges": edges, "ref_counts": counts}

    categorical: dict[str, dict[str, dict[str, int]]] = {}
    for col in CATEGORICAL_COLUMNS:
        if col not in X_train.columns:
            continue
        categorical[col] = {"ref_counts": _categorical_counts(X_train[col])}

    out_edges, out_counts = _numeric_bins(train_proba_positive, n_bins=n_bins)

    return {
        "n_samples": int(len(X_train)),
        "positive_rate": float(positive_rate),
        "computed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_bins": int(n_bins),
        "numeric": numeric,
        "categorical": categorical,
        "output": {"bin_edges": out_edges, "ref_counts": out_counts},
    }
