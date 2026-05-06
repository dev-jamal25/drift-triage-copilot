"""MLflow logging, registration, and load-back for the platform classifier.

Slice 2: produces the artifact triple — model binary + JSON schema +
filled-in model card — registers the model in the MLflow registry, applies
a `staging` alias, and verifies a load-back fidelity check (predictions
from the loaded model match the original within float tolerance).

MLflow 3 deprecates legacy stages in favour of aliases. The platform contract
keeps ``Staging`` / ``Production`` / ``Archived`` as external names; internally
we map them to lowercase aliases.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

logger = logging.getLogger("platform.ml.registry")

DEFAULT_MODEL_NAME = "bank-marketing-classifier"
DEFAULT_TRACKING_URI = "file:./mlruns"
STAGE_TO_ALIAS: dict[str, str] = {
    "Staging": "staging",
    "Production": "production",
    "Archived": "archived",
}


@dataclass(frozen=True)
class LoggedRun:
    run_id: str
    model_uri: str
    registered_name: str
    version: str
    alias: str


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 over the cleaned frame's bytes.

    Uses the parquet round-trip (column order + values + dtypes) so that
    re-running on the same source CSV produces the same hash.
    """
    payload = df.to_parquet(index=False)
    return hashlib.sha256(payload).hexdigest()


def compute_env_fingerprint() -> dict[str, str]:
    """Library + interpreter versions captured at training time."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "mlflow": mlflow.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def build_model_schema(pipe: Pipeline) -> dict[str, Any]:
    """Describe the model's input columns + output shape as a plain JSON dict.

    Categorical levels are pulled from the fitted ``OneHotEncoder``, so the
    schema reflects exactly what the model can accept without falling back to
    ``handle_unknown='ignore'``.
    """
    preprocessor = cast(ColumnTransformer, pipe.named_steps["preprocessor"])
    cat_encoder = cast(OneHotEncoder, preprocessor.named_transformers_["cat"])
    cat_levels: dict[str, list[str]] = {}
    for col, levels in zip(CATEGORICAL_COLUMNS, cat_encoder.categories_, strict=True):
        cat_levels[col] = [str(level) for level in levels]

    return {
        "input": {
            "numeric": list(NUMERIC_COLUMNS),
            "categorical": cat_levels,
        },
        "output": {
            "score": "float in [0, 1] — P(target == 1)",
            "label": "int 0 | 1 — 1 iff score >= threshold",
            "threshold": "float in [0, 1] — registered alongside the model",
        },
        "class_labels": {"0": "no", "1": "yes"},
    }


def render_model_card(
    template_path: Path,
    *,
    model_name: str,
    model_version: str,
    model_family: str,
    threshold: float,
    min_recall: float,
    alias: str,
    n_train: int,
    n_val: int,
    n_test: int,
    positive_rate: float,
    dataset_hash: str,
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    env_fingerprint: dict[str, str],
) -> str:
    """Render the model card markdown with concrete run values."""
    template = template_path.read_text(encoding="utf-8")
    env_md = "\n".join(f"- **{k}:** `{v}`" for k, v in sorted(env_fingerprint.items()))
    return template.format(
        model_name=model_name,
        model_version=model_version,
        model_family=model_family,
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        threshold=threshold,
        min_recall=min_recall,
        alias=alias,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        positive_rate=positive_rate,
        dataset_hash=dataset_hash,
        val_accuracy=val_metrics["accuracy"],
        val_f1_macro=val_metrics["f1_macro"],
        val_roc_auc=val_metrics["roc_auc"],
        val_precision=val_metrics["precision"],
        val_recall=val_metrics["recall"],
        test_accuracy=test_metrics["accuracy"],
        test_f1_macro=test_metrics["f1_macro"],
        test_roc_auc=test_metrics["roc_auc"],
        test_precision=test_metrics["precision"],
        test_recall=test_metrics["recall"],
        env_fingerprint_md=env_md,
    )


def log_and_register(
    pipe: Pipeline,
    *,
    model_name: str,
    model_family: str,
    threshold: float,
    min_recall: float,
    cleaned_df_for_hash: pd.DataFrame,
    sample_X: pd.DataFrame,
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    n_train: int,
    n_val: int,
    n_test: int,
    positive_rate: float,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = "bank-marketing",
    stage: str = "Staging",
    model_card_template: Path | None = None,
    extra_tags: dict[str, str] | None = None,
) -> LoggedRun:
    """One-shot: log model + schema + card, register, set alias, return URIs.

    Returns a ``LoggedRun`` whose ``model_uri`` is the alias-qualified URI
    suitable for ``mlflow.sklearn.load_model``.
    """
    if stage not in STAGE_TO_ALIAS:
        raise ValueError(f"Unknown stage '{stage}'. Use one of {list(STAGE_TO_ALIAS)}.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    dataset_hash = compute_dataset_hash(cleaned_df_for_hash)
    env_fingerprint = compute_env_fingerprint()
    schema = build_model_schema(pipe)

    sample = sample_X.head(5).copy()
    signature = infer_signature(sample, pipe.predict_proba(sample))

    template_path = (
        model_card_template
        if model_card_template is not None
        else Path(__file__).parent / "model_card.md"
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        mlflow.set_tags(
            {
                "dataset_hash": dataset_hash,
                "model_family": model_family,
                "stage_intent": stage,
                **{f"env.{k}": v for k, v in env_fingerprint.items()},
                **(extra_tags or {}),
            }
        )
        mlflow.log_params(
            {
                "model_family": model_family,
                "threshold": threshold,
                "min_recall": min_recall,
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
                "positive_rate": positive_rate,
            }
        )
        for split, metrics in (("val", val_metrics), ("test", test_metrics)):
            for metric_key in ("accuracy", "f1_macro", "roc_auc", "precision", "recall"):
                mlflow.log_metric(f"{split}_{metric_key}", float(metrics[metric_key]))

        mlflow.log_dict(schema, "schema.json")

        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            signature=signature,
            input_example=sample,
            registered_model_name=model_name,
        )

        client = MlflowClient()
        latest = max(
            client.search_model_versions(f"name='{model_name}'"),
            key=lambda mv: int(mv.version),
        )
        version = str(latest.version)

        rendered_card = render_model_card(
            template_path,
            model_name=model_name,
            model_version=version,
            model_family=model_family,
            threshold=threshold,
            min_recall=min_recall,
            alias=STAGE_TO_ALIAS[stage],
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            positive_rate=positive_rate,
            dataset_hash=dataset_hash,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            env_fingerprint=env_fingerprint,
        )
        mlflow.log_text(rendered_card, "model_card.md")

        alias = STAGE_TO_ALIAS[stage]
        client.set_registered_model_alias(name=model_name, alias=alias, version=version)

        try:
            client.update_registered_model(
                name=model_name,
                description="Drift Triage Co-Pilot — UCI Bank Marketing classifier.",
            )
        except Exception:
            logger.debug("registered model description not updated", exc_info=True)

    aliased_uri = f"models:/{model_name}@{alias}"
    logger.info(
        "registered model name=%s version=%s alias=%s run_id=%s",
        model_name,
        version,
        alias,
        run_id,
    )
    return LoggedRun(
        run_id=run_id,
        model_uri=aliased_uri,
        registered_name=model_name,
        version=version,
        alias=alias,
    )


def load_model(model_uri_or_alias: str) -> Pipeline:
    """Load a registered model. Accepts either a full URI or ``name@alias``."""
    if model_uri_or_alias.startswith("models:/"):
        uri = model_uri_or_alias
    elif "@" in model_uri_or_alias:
        uri = f"models:/{model_uri_or_alias}"
    else:
        raise ValueError(
            "Pass a 'models:/...' URI or 'name@alias' shorthand "
            "(e.g. 'bank-marketing-classifier@staging')."
        )
    return mlflow.sklearn.load_model(uri)


def fidelity_check(
    original: Pipeline,
    loaded: Pipeline,
    X_sample: pd.DataFrame,
    atol: float = 1e-9,
) -> dict[str, Any]:
    """Confirm the loaded pipeline reproduces the original's predict_proba.

    Raises ``AssertionError`` if any element exceeds ``atol``. Returns a small
    summary dict on success.
    """
    p_orig = original.predict_proba(X_sample)
    p_loaded = loaded.predict_proba(X_sample)
    diff = np.abs(p_orig - p_loaded)
    max_abs = float(diff.max())
    if max_abs > atol:
        raise AssertionError(f"Fidelity check failed: max|Δp|={max_abs} > atol={atol}.")
    return {
        "n_samples": int(len(X_sample)),
        "max_abs_diff": max_abs,
        "atol": atol,
    }


def write_run_summary(out_dir: Path, run: LoggedRun, fidelity: dict[str, Any]) -> Path:
    """Persist the registry summary next to model.joblib for easy inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": run.run_id,
        "model_uri": run.model_uri,
        "registered_name": run.registered_name,
        "version": run.version,
        "alias": run.alias,
        "fidelity_check": fidelity,
    }
    target = out_dir / "registry.json"
    target.write_text(json.dumps(summary, indent=2))
    return target
