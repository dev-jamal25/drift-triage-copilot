"""Slice 1 training entry point.

Reproducibly runs:
  load -> validate -> clean -> stratified 60/20/20 split
  -> majority baseline + 5-fold CV bake-off (logreg, rf, gb)
  -> pick best by mean f1_macro
  -> fit final pipeline on train, tune threshold on val (recall >= 0.75)
  -> single test-set evaluation
  -> persist train/val/test csv, metrics JSON, fitted pipeline (.joblib)

Run from the ``platform/`` directory:

    uv run python -m ml.train --data data/raw/bank-additional-full.csv --out data/processed/

No MLflow registration here — that lands in Slice 2.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from ml.data import (
    TARGET_COLUMN,
    clean,
    load_raw,
    split,
    split_xy,
    validate_columns,
)
from ml.evaluate import evaluate_at_threshold, tune_threshold
from ml.features import build_preprocessor
from ml.registry import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TRACKING_URI,
    LoggedRun,
    fidelity_check,
    load_model,
    log_and_register,
    write_run_summary,
)

logger = logging.getLogger("platform.ml.train")

CV_SCORERS = ("accuracy", "f1_macro", "roc_auc")
RANDOM_STATE = 42


def _candidate_estimators() -> dict[str, Any]:
    """Estimators included in the bake-off.

    Note: ``GradientBoostingClassifier`` does not support ``class_weight``;
    we lean on threshold tuning to handle imbalance for that model.
    """
    return {
        "logreg": LogisticRegression(
            class_weight="balanced", max_iter=2000, n_jobs=None, solver="lbfgs"
        ),
        "rf": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gb": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def build_pipeline(model_name: str) -> Pipeline:
    """Compose preprocessor + estimator into a single sklearn Pipeline."""
    estimators = _candidate_estimators()
    if model_name not in estimators:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {sorted(estimators)}.")
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimators[model_name]),
        ]
    )


def majority_baseline_scores(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
    """Score a most-frequent-class dummy classifier with the same CV split."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]
    )
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=list(CV_SCORERS), n_jobs=-1)
    return {
        "accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "accuracy_std": float(np.std(scores["test_accuracy"])),
        "f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
        "f1_macro_std": float(np.std(scores["test_f1_macro"])),
        "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "roc_auc_std": float(np.std(scores["test_roc_auc"])),
    }


def cv_compare(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """5-fold stratified CV across all candidate estimators. Returns mean ± std table."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, float | str]] = []
    for name in _candidate_estimators():
        pipe = build_pipeline(name)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=list(CV_SCORERS), n_jobs=-1)
        rows.append(
            {
                "model": name,
                "accuracy_mean": float(np.mean(scores["test_accuracy"])),
                "accuracy_std": float(np.std(scores["test_accuracy"])),
                "f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
                "f1_macro_std": float(np.std(scores["test_f1_macro"])),
                "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
                "roc_auc_std": float(np.std(scores["test_roc_auc"])),
            }
        )
    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)


def fit_final(model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit the chosen pipeline on the full training set."""
    pipe = build_pipeline(model_name)
    pipe.fit(X_train, y_train)
    return pipe


@dataclass(frozen=True)
class TrainResult:
    chosen_model: str
    threshold: float
    cv_table: pd.DataFrame
    baseline: dict[str, float]
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    registry: LoggedRun | None = None
    fidelity: dict[str, Any] | None = None


def run_pipeline(
    data_path: Path,
    out_dir: Path,
    model_name: str = "auto",
    min_recall: float = 0.75,
    register: bool = True,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = "bank-marketing",
    registered_model_name: str = DEFAULT_MODEL_NAME,
    stage: str = "Staging",
) -> TrainResult:
    """End-to-end Slice 1+2 pipeline. Returns metrics; persists artifacts to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading dataset path=%s", data_path)
    raw = load_raw(data_path)
    validate_columns(raw)
    cleaned = clean(raw)

    train_df, val_df, test_df = split(cleaned, random_state=RANDOM_STATE)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)

    logger.info("scoring majority baseline")
    baseline = majority_baseline_scores(X_train, y_train)

    logger.info("running 5-fold CV bake-off")
    cv_table = cv_compare(X_train, y_train)

    chosen = (
        cv_table.iloc[0]["model"] if model_name == "auto" else model_name  # type: ignore[index]
    )
    logger.info("fitting final pipeline model=%s", chosen)
    pipe = fit_final(str(chosen), X_train, y_train)

    val_proba = pipe.predict_proba(X_val)[:, 1]
    threshold = tune_threshold(y_val.to_numpy(), val_proba, min_recall=min_recall)
    val_metrics = evaluate_at_threshold(y_val.to_numpy(), val_proba, threshold)

    test_proba = pipe.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_at_threshold(y_test.to_numpy(), test_proba, threshold)

    joblib.dump(pipe, out_dir / "model.joblib")
    metrics_payload: dict[str, Any] = {
        "chosen_model": str(chosen),
        "threshold": threshold,
        "min_recall": min_recall,
        "baseline": baseline,
        "cv_table": cv_table.to_dict(orient="records"),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "positive_rate": float(cleaned[TARGET_COLUMN].mean()),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    registry: LoggedRun | None = None
    fidelity: dict[str, Any] | None = None
    if register:
        logger.info(
            "registering model in MLflow tracking_uri=%s name=%s stage=%s",
            tracking_uri,
            registered_model_name,
            stage,
        )
        registry = log_and_register(
            pipe,
            model_name=registered_model_name,
            model_family=str(chosen),
            threshold=threshold,
            min_recall=min_recall,
            cleaned_df_for_hash=cleaned,
            sample_X=X_val,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            n_train=int(len(train_df)),
            n_val=int(len(val_df)),
            n_test=int(len(test_df)),
            positive_rate=float(cleaned[TARGET_COLUMN].mean()),
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            stage=stage,
        )
        loaded = load_model(registry.model_uri)
        fidelity = fidelity_check(pipe, loaded, X_val.head(200))
        write_run_summary(out_dir, registry, fidelity)

    return TrainResult(
        chosen_model=str(chosen),
        threshold=threshold,
        cv_table=cv_table,
        baseline=baseline,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        registry=registry,
        fidelity=fidelity,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Bank Marketing classifier (Slice 1).")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/bank-additional-full.csv"),
        help="Path to the UCI bank-additional-full.csv (semicolon-separated).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/"),
        help="Directory for processed splits, model.joblib, and metrics.json.",
    )
    parser.add_argument(
        "--model",
        choices=("auto", "logreg", "rf", "gb"),
        default="auto",
        help="'auto' picks the best model from CV by mean f1_macro.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.75,
        help="Recall floor for threshold tuning.",
    )
    parser.add_argument(
        "--register",
        dest="register",
        action="store_true",
        help="Register the model in MLflow (default).",
    )
    parser.add_argument(
        "--no-register",
        dest="register",
        action="store_false",
        help="Skip MLflow logging and registration.",
    )
    parser.set_defaults(register=True)
    parser.add_argument(
        "--mlflow-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI (e.g. file:./mlruns or http://mlflow:5000).",
    )
    parser.add_argument(
        "--experiment",
        default="bank-marketing",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--registered-name",
        default=DEFAULT_MODEL_NAME,
        help="Registered model name in the MLflow registry.",
    )
    parser.add_argument(
        "--stage",
        choices=("Staging", "Production", "Archived"),
        default="Staging",
        help="Stage to assign as an alias on the new version.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    if not args.data.exists():
        logger.error("dataset not found at %s — place bank-additional-full.csv there.", args.data)
        return 2

    result = run_pipeline(
        data_path=args.data,
        out_dir=args.out,
        model_name=args.model,
        min_recall=args.min_recall,
        register=args.register,
        tracking_uri=args.mlflow_uri,
        experiment_name=args.experiment,
        registered_model_name=args.registered_name,
        stage=args.stage,
    )

    print("\n=== CV bake-off (mean ± std, sorted by f1_macro) ===")
    print(result.cv_table.to_string(index=False))
    print("\n=== Majority-class baseline (5-fold CV on train) ===")
    print(json.dumps(result.baseline, indent=2))
    print(f"\nChosen model: {result.chosen_model}")
    print(f"Tuned threshold (recall >= {args.min_recall} on val): {result.threshold:.4f}")
    print("\n=== Validation metrics @ tuned threshold ===")
    print(json.dumps(result.val_metrics, indent=2))
    print("\n=== Test metrics @ tuned threshold (single shot) ===")
    print(json.dumps(result.test_metrics, indent=2))

    if result.registry is not None:
        print("\n=== MLflow registry ===")
        print(f"  run_id:    {result.registry.run_id}")
        print(f"  model_uri: {result.registry.model_uri}")
        print(f"  version:   {result.registry.version}")
        print(f"  alias:     {result.registry.alias}")
        if result.fidelity is not None:
            print(f"  fidelity:  {result.fidelity}")

    print(f"\nArtifacts: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
