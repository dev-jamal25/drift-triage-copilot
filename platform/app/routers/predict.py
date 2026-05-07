"""Async ``POST /predict`` route."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Annotated

import pandas as pd
import structlog
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_model_bundle, get_session
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.model_loader import ModelBundle
from app.services.prediction_log import insert_prediction
from ml.data import apply_pdays_sentinel

log = structlog.get_logger(__name__)

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    req: PredictionRequest,
    bundle: Annotated[ModelBundle, Depends(get_model_bundle)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PredictionResponse:
    """Score a single client and persist the row to ``predictions_log``."""
    raw = req.model_dump(by_alias=True)
    feature_df = apply_pdays_sentinel(pd.DataFrame([raw]))

    proba = await asyncio.to_thread(bundle.pipeline.predict_proba, feature_df)
    score = float(proba[0, 1])
    label = 1 if score >= bundle.threshold else 0
    predicted_at = datetime.now(UTC)

    await insert_prediction(
        session,
        predicted_at=predicted_at,
        model_name=bundle.model_name,
        model_version=bundle.version,
        threshold=bundle.threshold,
        score=score,
        label=label,
        features=raw,
    )
    await session.commit()

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
        predicted_at=predicted_at,
    )
