"""Assemble a live window from ``predictions_log`` and compute a DriftReport.

Pure orchestration over the Step 3 ``compute_drift_report`` and Step 1
``prediction_log.fetch_recent``. The scheduler calls this once per tick.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import structlog
from shared.contracts import DriftReport
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.model_loader import ModelBundle
from app.services.prediction_log import fetch_recent
from ml.data import apply_pdays_sentinel
from ml.drift_metrics import compute_drift_report

log = structlog.get_logger(__name__)


def _empty_report() -> DriftReport:
    now = datetime.now(UTC)
    return DriftReport(
        window_start=now,
        window_end=now,
        sample_size=0,
        overall_severity="green",
        feature_drifts=[],
        output_drift_psi=0.0,
    )


async def recompute_drift(
    session: AsyncSession,
    bundle: ModelBundle,
    window_size: int,
) -> DriftReport:
    """Compute drift over the last ``window_size`` rows for the served model.

    Rows from older model versions are filtered out — the reference stats
    on ``bundle`` were computed for *this* model only, so mixing versions
    would produce a meaningless comparison.
    """
    if bundle.reference_stats is None:
        raise RuntimeError(
            "recompute_drift called with a bundle that has no reference_stats. "
            "Re-train + re-register the model with Step 2's pipeline."
        )

    rows = await fetch_recent(session, window_size)
    matching = [
        r for r in rows if r.model_name == bundle.model_name and r.model_version == bundle.version
    ]
    if not matching:
        log.info(
            "drift_recompute_no_rows",
            model_name=bundle.model_name,
            model_version=bundle.version,
            window_size=window_size,
        )
        return _empty_report()

    raw_df = pd.DataFrame([r.features for r in matching])
    cleaned_df = apply_pdays_sentinel(raw_df)
    proba = np.array([float(r.score) for r in matching], dtype=float)

    window_start = min(r.predicted_at for r in matching)
    window_end = max(r.predicted_at for r in matching)

    report = compute_drift_report(
        bundle.reference_stats,
        cleaned_df,
        proba,
        window_start=window_start,
        window_end=window_end,
    )
    log.info(
        "drift_recompute_ok",
        model_name=bundle.model_name,
        model_version=bundle.version,
        sample_size=report.sample_size,
        overall_severity=report.overall_severity,
        output_psi=report.output_drift_psi,
    )
    return report
