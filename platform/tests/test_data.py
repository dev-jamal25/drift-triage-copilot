"""Tests for the dataset cleaning and split rules.

Uses a small synthetic frame matching the UCI Bank Marketing schema. CI must
not depend on the real CSV being present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.data import (
    CATEGORICAL_COLUMNS,
    EXPECTED_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    clean,
    split,
    validate_columns,
)


def _synthetic_raw(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Build a UCI-shaped synthetic frame with the right columns and dtypes."""
    rng = np.random.default_rng(seed)
    job_choices = ["admin.", "blue-collar", "technician", "services", "unknown"]
    return pd.DataFrame(
        {
            "age": rng.integers(18, 95, size=n),
            "job": rng.choice(job_choices, size=n),
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
            # ~70% of pdays are sentinel, mirroring the real dataset's skew
            "pdays": np.where(rng.random(n) < 0.7, 999, rng.integers(0, 30, size=n)),
            "previous": rng.integers(0, 5, size=n),
            "poutcome": rng.choice(["success", "failure", "nonexistent"], size=n),
            "emp.var.rate": rng.uniform(-3, 2, size=n),
            "cons.price.idx": rng.uniform(92, 95, size=n),
            "cons.conf.idx": rng.uniform(-50, -25, size=n),
            "euribor3m": rng.uniform(0.5, 5.0, size=n),
            "nr.employed": rng.uniform(4900, 5230, size=n),
            "y": rng.choice(["yes", "no"], size=n, p=[0.11, 0.89]),
        }
    )


def test_validate_columns_accepts_canonical_schema() -> None:
    df = _synthetic_raw(50)
    assert tuple(df.columns) == EXPECTED_COLUMNS
    validate_columns(df)


def test_validate_columns_rejects_missing_column() -> None:
    df = _synthetic_raw(50).drop(columns=["duration"])
    with pytest.raises(ValueError, match="Column mismatch"):
        validate_columns(df)


def test_duration_dropped() -> None:
    cleaned = clean(_synthetic_raw(200))
    assert "duration" not in cleaned.columns


def test_target_is_binary_int() -> None:
    cleaned = clean(_synthetic_raw(500))
    assert TARGET_COLUMN in cleaned.columns
    assert "y" not in cleaned.columns
    assert set(cleaned[TARGET_COLUMN].unique()).issubset({0, 1})
    assert pd.api.types.is_integer_dtype(cleaned[TARGET_COLUMN])


def test_pdays_sentinel_features_exist() -> None:
    cleaned = clean(_synthetic_raw(500))
    assert "pdays" not in cleaned.columns
    assert "pdays_clean" in cleaned.columns
    assert "was_previously_contacted" in cleaned.columns
    # was_previously_contacted is a 0/1 flag
    assert set(cleaned["was_previously_contacted"].unique()).issubset({0, 1})
    # pdays_clean has zero-valued rows wherever the flag is 0
    assert (cleaned.loc[cleaned["was_previously_contacted"] == 0, "pdays_clean"] == 0).all()


def test_unknown_preserved_as_category() -> None:
    raw = _synthetic_raw(500)
    # force at least one "unknown" in a categorical we expect to retain it
    raw.loc[0, "job"] = "unknown"
    cleaned = clean(raw)
    for col in CATEGORICAL_COLUMNS:
        assert col in cleaned.columns
    assert "unknown" in cleaned["job"].unique()


def test_split_proportions_60_20_20() -> None:
    cleaned = clean(_synthetic_raw(10_000, seed=1))
    train_df, val_df, test_df = split(cleaned, random_state=42)
    total = len(cleaned)
    assert len(train_df) + len(val_df) + len(test_df) == total
    # each share within ±1pp of the target
    assert abs(len(train_df) / total - 0.60) < 0.01
    assert abs(len(val_df) / total - 0.20) < 0.01
    assert abs(len(test_df) / total - 0.20) < 0.01


def test_class_balance_preserved() -> None:
    cleaned = clean(_synthetic_raw(10_000, seed=2))
    overall = cleaned[TARGET_COLUMN].mean()
    train_df, val_df, test_df = split(cleaned, random_state=42)
    for part in (train_df, val_df, test_df):
        assert abs(part[TARGET_COLUMN].mean() - overall) < 0.005


def test_split_is_deterministic() -> None:
    cleaned = clean(_synthetic_raw(2_000, seed=3))
    a = split(cleaned, random_state=42)
    b = split(cleaned, random_state=42)
    for left, right in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(left, right)


def test_numeric_columns_present_after_clean() -> None:
    cleaned = clean(_synthetic_raw(200))
    for col in NUMERIC_COLUMNS:
        assert col in cleaned.columns
