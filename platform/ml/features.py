"""Feature preprocessing: scaler for numerics, one-hot for categoricals.

``"unknown"`` survives as a literal category — `OneHotEncoder` treats it as a
level, so no special handling is needed.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    """Return a fresh ColumnTransformer for the cleaned feature set."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC_COLUMNS)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_COLUMNS),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
