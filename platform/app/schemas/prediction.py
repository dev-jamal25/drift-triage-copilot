"""Platform-internal prediction request/response schemas.

These are deliberately *not* in ``shared/contracts.py``: the predict API is
not a wire contract between platform and agent — it's the public boundary of
the model service. ``shared/contracts.py`` is only for cross-service contracts
(DriftEvent, PromotionRequest, etc.).

Field aliases match the raw UCI column names (with dots) so a request body
mirrors a row from the training CSV. ``populate_by_name=True`` also lets
clients use snake_case if they prefer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """One client at one point in time. Excludes ``duration`` (target leak)
    and the label ``y``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    age: int = Field(..., ge=17, le=120)
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int = Field(..., ge=1, description="# contacts during this campaign")
    pdays: int = Field(
        ...,
        ge=0,
        le=999,
        description="Days since last contact in a previous campaign; 999 = never.",
    )
    previous: int = Field(..., ge=0)
    poutcome: str
    emp_var_rate: float = Field(..., alias="emp.var.rate")
    cons_price_idx: float = Field(..., alias="cons.price.idx")
    cons_conf_idx: float = Field(..., alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(..., alias="nr.employed")


class PredictionResponse(BaseModel):
    """Score + decision + provenance."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(..., ge=0.0, le=1.0, description="P(target == 1)")
    label: Literal[0, 1]
    threshold: float = Field(..., ge=0.0, le=1.0)
    model_name: str
    model_version: str
    predicted_at: datetime
