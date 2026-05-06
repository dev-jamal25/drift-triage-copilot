"""Async ``POST /predict`` route."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Annotated

import pandas as pd
import structlog
from fastapi import APIRouter, Depends

from app.dependencies import get_model_bundle
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.model_loader import ModelBundle
from ml.data import apply_pdays_sentinel

log = structlog.get_logger(__name__)

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    req: PredictionRequest,
    bundle: Annotated[ModelBundle, Depends(get_model_bundle)],
) -> PredictionResponse:
    """Score a single client. Threshold and model version come from the bundle."""
    raw = req.model_dump(by_alias=True)
    feature_df = apply_pdays_sentinel(pd.DataFrame([raw]))

    proba = await asyncio.to_thread(bundle.pipeline.predict_proba, feature_df)
    score = float(proba[0, 1])
    label = 1 if score >= bundle.threshold else 0

    log.info(
        "predict",
        model_name=bundle.model_name,
        model_version=bundle.version,
        score=score,
        label=label,
    )

    return PredictionResponse(
        score=score,
        label=label,
        threshold=bundle.threshold,
        model_name=bundle.model_name,
        model_version=bundle.version,
        predicted_at=datetime.now(UTC),
    )
