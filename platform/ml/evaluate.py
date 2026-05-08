"""Threshold tuning and metric evaluation.

Threshold rule: pick the **highest** decision threshold whose recall on the
validation set is at least ``min_recall`` (default 0.75). Choosing the highest
qualifying threshold maximises precision subject to the recall floor — fewer
false alarms while still catching enough positives.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def tune_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_proba: Sequence[float] | np.ndarray,
    min_recall: float = 0.75,
) -> float:
    """Return the highest threshold whose recall on (y_true, y_proba) is >= min_recall.

    Raises ValueError if no threshold in the sweep achieves the floor.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba, dtype=float)
    if y_true_arr.shape != y_proba_arr.shape:
        raise ValueError("y_true and y_proba must have the same shape.")
    if not (0.0 <= min_recall <= 1.0):
        raise ValueError("min_recall must be in [0, 1].")

    candidates = np.linspace(0.01, 0.99, 99)
    qualifying: list[float] = []
    for t in candidates:
        preds = (y_proba_arr >= t).astype(int)
        recall = recall_score(y_true_arr, preds, zero_division=0)
        if recall >= min_recall:
            qualifying.append(float(t))

    if not qualifying:
        raise ValueError(f"No threshold in [0.01, 0.99] achieves recall >= {min_recall}.")
    return max(qualifying)


def evaluate_at_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_proba: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict[str, float | list[list[int]]]:
    """Compute accuracy, macro F1, ROC-AUC, precision, recall, confusion matrix."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba, dtype=float)
    preds = (y_proba_arr >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_arr, preds)),
        "f1_macro": float(f1_score(y_true_arr, preds, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_proba_arr)),
        "precision": float(precision_score(y_true_arr, preds, zero_division=0)),
        "recall": float(recall_score(y_true_arr, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_arr, preds).tolist(),
    }
