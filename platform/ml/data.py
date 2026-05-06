"""Dataset loading, validation, cleaning, and stratified splitting.

All functions are pure and deterministic. The only I/O is `load_raw`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EXPECTED_COLUMNS: tuple[str, ...] = (
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "y",
)

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "age",
    "campaign",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "was_previously_contacted",
    "pdays_clean",
)

TARGET_COLUMN = "target"
PDAYS_SENTINEL = 999


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the UCI Bank Marketing CSV (semicolon-separated)."""
    return pd.read_csv(path, sep=";")


def validate_columns(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if the frame's columns do not match the UCI schema exactly."""
    actual = tuple(df.columns)
    if actual != EXPECTED_COLUMNS:
        missing = set(EXPECTED_COLUMNS) - set(actual)
        extra = set(actual) - set(EXPECTED_COLUMNS)
        raise ValueError(f"Column mismatch. missing={sorted(missing)} extra={sorted(extra)}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the platform's data-prep rules.

    Rules locked by Slice 1 tests:
      * drop ``duration`` (target leakage)
      * map ``y`` -> ``target`` int {0, 1}
      * keep ``"unknown"`` as a literal category (no fillna/replace)
      * encode ``pdays == 999`` as a sentinel via ``was_previously_contacted``
        and ``pdays_clean``; drop original ``pdays``
    """
    out = df.copy()

    out = out.drop(columns=["duration"])

    out[TARGET_COLUMN] = out["y"].map({"yes": 1, "no": 0}).astype("int64")
    out = out.drop(columns=["y"])

    out["was_previously_contacted"] = (out["pdays"] != PDAYS_SENTINEL).astype("int64")
    out["pdays_clean"] = np.where(out["pdays"] == PDAYS_SENTINEL, 0, out["pdays"]).astype("int64")
    out = out.drop(columns=["pdays"])

    return out


def split(
    df: pd.DataFrame, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 60/20/20 split on ``target``.

    Two ``train_test_split`` passes: 60/40, then 50/50 on the held-out 40%.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"split() expects a cleaned frame with '{TARGET_COLUMN}' column.")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.4,
        stratify=df[TARGET_COLUMN],
        random_state=random_state,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df[TARGET_COLUMN],
        random_state=random_state,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) from the target column (y)."""
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]
