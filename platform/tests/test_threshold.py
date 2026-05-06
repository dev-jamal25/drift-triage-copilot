"""Tests for the threshold tuning rule."""

from __future__ import annotations

import numpy as np
import pytest

from ml.evaluate import evaluate_at_threshold, tune_threshold


def test_threshold_rule_picks_highest_at_min_recall() -> None:
    # 10 positives at [0.10, 0.20, ..., 1.00]; 10 negatives at 0.00
    # At t=0.75: predicts positives only for proba >= 0.75 -> 3 of 10 captured (recall 0.3)
    # At t=0.30: 7 of 10 captured (recall 0.7)
    # At t=0.20: 9 of 10 captured (recall 0.9) — passes 0.75 floor
    # At t=0.10: 10 of 10 captured (recall 1.0) — also passes
    # Want the *highest* qualifying threshold -> 0.20 (or nearest sweep point >=0.20)
    pos_probs = np.linspace(0.10, 1.00, 10)
    neg_probs = np.zeros(10)
    y = np.array([1] * 10 + [0] * 10)
    proba = np.concatenate([pos_probs, neg_probs])

    t = tune_threshold(y, proba, min_recall=0.75)
    # Sweep is 0.01..0.99 step ~0.01 — assert we land on the highest threshold
    # for which 8 of 10 positives are still >= t (recall 0.8 satisfies the floor).
    # Positives at 0.30..1.00 are 8 samples; threshold can be as high as 0.30.
    assert t == pytest.approx(0.30, abs=0.02)


def test_threshold_rule_raises_when_no_threshold_meets_recall() -> None:
    # Probabilities so low that no positive prediction can meet recall>=0.75.
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    proba = np.array([0.005, 0.005, 0.005, 0.005, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="No threshold"):
        tune_threshold(y, proba, min_recall=0.75)


def test_threshold_rule_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        tune_threshold([0, 1, 1], [0.1, 0.2], min_recall=0.5)


def test_threshold_rule_rejects_invalid_min_recall() -> None:
    with pytest.raises(ValueError):
        tune_threshold([0, 1], [0.1, 0.9], min_recall=1.5)


def test_evaluate_at_threshold_returns_full_metric_set() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    proba = rng.random(200)
    out = evaluate_at_threshold(y, proba, threshold=0.5)
    for key in (
        "threshold",
        "accuracy",
        "f1_macro",
        "roc_auc",
        "precision",
        "recall",
        "confusion_matrix",
    ):
        assert key in out
    assert out["threshold"] == 0.5
    cm = out["confusion_matrix"]
    assert isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2
