"""Bootstrap: train and register model in MLflow for a clean Docker startup.

Runs only once on compose up, before model-service starts.
Idempotent: if model@alias already exists, skips training.
"""

import sys
from pathlib import Path

import structlog

from ml.registry import get_registered_model_uri
from ml.train import main as train_main

log = structlog.get_logger("bootstrap")


def main() -> int:
    """Train and register the bank-marketing-classifier model.

    Returns 0 on success, 1 on failure.
    """
    tracking_uri = "http://mlflow:5000"
    model_name = "bank-marketing-classifier"
    model_alias = "staging"
    data_path = Path(__file__).parent / "data" / "raw" / "bank-additional-full.csv"
    out_dir = Path(__file__).parent / "data" / "processed"

    if not data_path.exists():
        log.error("data_file_missing", path=str(data_path))
        return 1

    log.info(
        "bootstrap_start",
        model_name=model_name,
        model_alias=model_alias,
        mlflow_uri=tracking_uri,
    )

    # Check if model@alias already exists
    existing_uri = get_registered_model_uri(
        tracking_uri=tracking_uri,
        model_name=model_name,
        alias=model_alias,
    )
    if existing_uri:
        log.info(
            "model_exists_skipping_train",
            model_uri=existing_uri,
        )
        return 0

    # Train and register via CLI
    try:
        argv = [
            "--data",
            str(data_path),
            "--out",
            str(out_dir),
            "--register",
            "--mlflow-uri",
            tracking_uri,
            "--experiment",
            "bank-marketing-platform",
            "--registered-name",
            model_name,
            "--stage",
            model_alias,
        ]
        result = train_main(argv)
        if result == 0:
            log.info("bootstrap_complete", model_name=model_name, alias=model_alias)
        return result
    except Exception as e:
        log.exception("bootstrap_failed", error=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
