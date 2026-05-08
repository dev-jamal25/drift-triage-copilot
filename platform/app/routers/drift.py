"""``GET /drift`` — return the latest cached drift report."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_drift_scheduler, get_model_bundle
from app.schemas.drift import DriftReportResponse
from app.services.drift_scheduler import DriftScheduler
from app.services.model_loader import ModelBundle

router = APIRouter(tags=["drift"])


@router.get("/drift", response_model=DriftReportResponse)
async def drift(
    scheduler: Annotated[DriftScheduler, Depends(get_drift_scheduler)],
    bundle: Annotated[ModelBundle, Depends(get_model_bundle)],
) -> DriftReportResponse:
    """Return the latest drift report, or 503 if no tick has produced one yet."""
    if scheduler.latest_report is None or scheduler.last_refreshed_at is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Drift not yet computed.",
        )
    return DriftReportResponse(
        drift=scheduler.latest_report,
        model_name=bundle.model_name,
        model_version=bundle.version,
        last_refreshed_at=scheduler.last_refreshed_at,
    )
