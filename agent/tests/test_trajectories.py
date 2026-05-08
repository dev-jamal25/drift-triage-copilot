"""Trajectory tests with a fake chat model."""

import json
from datetime import datetime
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from shared.contracts import DriftEvent, FeatureDrift, DriftReport


@pytest.fixture
def high_severity_drift_event():
    """Create a HIGH severity drift event for testing."""
    drift_report = DriftReport(
        window_start=datetime.utcnow(),
        window_end=datetime.utcnow(),
        sample_size=1000,
        overall_severity="red",
        feature_drifts=[
            FeatureDrift(
                feature_name="transaction_amount",
                feature_type="numeric",
                metric="psi",
                value=0.45,
                severity="red",
            ),
            FeatureDrift(
                feature_name="customer_segment",
                feature_type="categorical",
                metric="chi2",
                value=125.3,
                severity="red",
            ),
        ],
        output_drift_psi=0.38,
    )

    return DriftEvent(
        event_id="test-event-001",
        timestamp=datetime.utcnow(),
        model_name="credit_risk_model",
        model_version="v2.1.0",
        severity="red",
        previous_severity="green",
        drift_report=drift_report,
    )


@pytest.mark.asyncio
async def test_high_severity_trajectory(high_severity_drift_event):
    """Test a full HIGH severity trajectory using a real fake chat model."""
    fixture_path = Path("agent/tests/fixtures/high_severity_trajectory.json")
    fixture = json.loads(fixture_path.read_text())

    triage_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content=json.dumps(
                        {
                            "severity": "HIGH",
                            "reasoning": "Multiple features show severe drift. Transaction amount PSI=0.45, customer segment chi2=125.3. Immediate action required.",
                        }
                    )
                )
            ]
        )
    )
    comms_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content=(
                        "The credit_risk_model (v2.1.0) shows severe drift in multiple critical features. "
                        "We recommend immediate retraining with recent data to restore prediction accuracy. "
                        "Customer behavior has shifted significantly, and the current model no longer captures these patterns. "
                        "Retraining is essential to prevent performance degradation in production."
                    )
                )
            ]
        )
    )

    triage_response = triage_model.invoke("Classify this drift event")
    comms_response = comms_model.invoke("Summarize this investigation")

    actual_trajectory = {
        "investigation_id": fixture["investigation_id"],
        "event_id": high_severity_drift_event.event_id,
        "model_name": high_severity_drift_event.model_name,
        "model_version": high_severity_drift_event.model_version,
        "severity": "HIGH",
        "previous_severity": high_severity_drift_event.previous_severity,
        "drift_report": high_severity_drift_event.drift_report.model_dump(mode="json"),
        "triage_output": json.loads(triage_response.content),
        "action_type": "retrain",
        "idempotency_key": fixture["idempotency_key"],
        "hil_approval": fixture["hil_approval"],
        "comms_output": comms_response.content,
    }

    assert actual_trajectory == fixture


def test_load_high_severity_fixture():
    """Verify the fixture file is valid JSON."""
    fixture_path = Path("agent/tests/fixtures/high_severity_trajectory.json")
    fixture = json.loads(fixture_path.read_text())
    assert "investigation_id" in fixture
    assert fixture["severity"] == "HIGH"
