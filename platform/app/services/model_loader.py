"""Load the registered classifier from MLflow at service startup.

The startup load is wrapped in a tenacity retry so a slow MLflow backend
doesn't kill the pod on first attempt. Per CLAUDE.md hard rules, every
external call has timeout + retry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import mlflow
import structlog
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from tenacity import (
    after_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import Settings

log = structlog.get_logger(__name__)

# Used for tenacity's stdlib-logger hook
import logging  # noqa: E402

_stdlog = logging.getLogger("app.services.model_loader")


@dataclass(frozen=True)
class ModelBundle:
    """Everything the request path needs about the loaded model.

    ``reference_stats`` is the JSON dict produced by
    ``ml.drift_stats.compute_reference_stats`` and logged at registration
    time. It's optional so we can still load older models registered before
    Step 2 — the drift scheduler will simply skip those runs.
    """

    pipeline: Pipeline
    threshold: float
    model_uri: str
    model_name: str
    version: str
    alias: str
    run_id: str
    loaded_at: datetime
    reference_stats: dict[str, Any] | None = None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
    reraise=True,
    after=after_log(_stdlog, logging.WARNING),
)
def load_bundle(settings: Settings) -> ModelBundle:
    """Resolve the alias, load the sklearn pipeline, and pull the threshold.

    The threshold is read from the originating MLflow run's params (where
    ``ml/registry.py`` logged it) so the registry is the single source of
    truth — there is no separate threshold file to drift out of sync.
    """
    uri = f"models:/{settings.model_name}@{settings.model_alias}"
    log.info("model_load_start", uri=uri, tracking_uri=settings.mlflow_tracking_uri)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    mv = client.get_model_version_by_alias(settings.model_name, settings.model_alias)
    pipe = mlflow.sklearn.load_model(uri)
    run = client.get_run(mv.run_id)
    if "threshold" not in run.data.params:
        raise RuntimeError(
            f"Run {mv.run_id} for {settings.model_name} v{mv.version} is missing "
            "a 'threshold' param — was it logged via ml/registry.py?"
        )
    threshold = float(run.data.params["threshold"])
    reference_stats = _load_reference_stats(run.info.run_id)

    bundle = ModelBundle(
        pipeline=pipe,
        threshold=threshold,
        model_uri=uri,
        model_name=settings.model_name,
        version=str(mv.version),
        alias=settings.model_alias,
        run_id=mv.run_id,
        loaded_at=datetime.now(UTC),
        reference_stats=reference_stats,
    )
    log.info(
        "model_load_ok",
        uri=uri,
        version=bundle.version,
        threshold=bundle.threshold,
        run_id=bundle.run_id,
        reference_stats=bool(reference_stats),
    )
    return bundle


def _load_reference_stats(run_id: str) -> dict[str, Any] | None:
    """Best-effort fetch of ``reference_stats.json`` from the model's run.

    Older runs (registered before Step 2) won't have the artifact. The drift
    scheduler treats ``None`` as "no baseline available — skip drift" rather
    than failing the whole service.
    """
    artifact_uri = f"runs:/{run_id}/reference_stats.json"
    try:
        return dict(mlflow.artifacts.load_dict(artifact_uri))
    except (MlflowException, FileNotFoundError, OSError) as exc:
        log.warning(
            "reference_stats_missing",
            run_id=run_id,
            reason=str(exc),
        )
        return None
