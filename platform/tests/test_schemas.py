"""Pydantic-only tests for prediction request/response schemas."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.schemas.prediction import PredictionRequest, PredictionResponse


def _canonical_payload() -> dict[str, object]:
    """A request body matching the UCI raw column names (with dots)."""
    return {
        "age": 35,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "thu",
        "campaign": 2,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": -1.8,
        "cons.price.idx": 93.075,
        "cons.conf.idx": -47.1,
        "euribor3m": 1.405,
        "nr.employed": 5099.1,
    }


def test_request_accepts_canonical_payload() -> None:
    req = PredictionRequest(**_canonical_payload())
    assert req.age == 35
    assert req.pdays == 999
    # by_alias dump round-trips back to dataset-style names for the model
    dumped = req.model_dump(by_alias=True)
    assert "emp.var.rate" in dumped
    assert "cons.price.idx" in dumped
    assert "nr.employed" in dumped


def test_request_accepts_snake_case_alternative() -> None:
    payload = _canonical_payload()
    payload["emp_var_rate"] = payload.pop("emp.var.rate")
    payload["cons_price_idx"] = payload.pop("cons.price.idx")
    payload["cons_conf_idx"] = payload.pop("cons.conf.idx")
    payload["nr_employed"] = payload.pop("nr.employed")
    req = PredictionRequest(**payload)
    assert req.emp_var_rate == -1.8


def test_request_rejects_extra_fields() -> None:
    payload = _canonical_payload()
    payload["duration"] = 100  # leakage column — must be rejected
    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


def test_request_rejects_missing_field() -> None:
    payload = _canonical_payload()
    del payload["age"]
    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


def test_request_rejects_pdays_above_sentinel() -> None:
    payload = _canonical_payload()
    payload["pdays"] = 1000
    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


def test_request_rejects_negative_age() -> None:
    payload = _canonical_payload()
    payload["age"] = -1
    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


def test_request_rejects_zero_campaign() -> None:
    payload = _canonical_payload()
    payload["campaign"] = 0
    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


def test_response_constrains_score_to_unit_interval() -> None:
    with pytest.raises(ValidationError):
        PredictionResponse(
            score=1.5,
            label=1,
            threshold=0.5,
            model_name="m",
            model_version="1",
            predicted_at=datetime.now(UTC),
        )


def test_response_label_must_be_zero_or_one() -> None:
    with pytest.raises(ValidationError):
        PredictionResponse(
            score=0.5,
            label=2,  # type: ignore[arg-type]
            threshold=0.5,
            model_name="m",
            model_version="1",
            predicted_at=datetime.now(UTC),
        )
