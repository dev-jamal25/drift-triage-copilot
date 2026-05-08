"""Triage node: classify drift severity from DriftEvent."""

import json
import os
from typing import Any

import structlog

log = structlog.get_logger()

TRIAGE_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "prompts", "triage.txt"
)


def load_triage_prompt() -> str:
    """Load the triage prompt from file."""
    with open(TRIAGE_PROMPT_PATH, "r") as f:
        return f.read()


def triage_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Triage node: classify drift severity from DriftEvent into LOW / MEDIUM / HIGH.

    For now, classify based on the drift_event's overall_severity.
    LLM integration will be done in a separate async invoke step.
    """
    log.info("triage.execute", investigation_id=state.get("investigation_id"))

    drift_event_dict = state.get("drift_event")
    if not drift_event_dict:
        log.error("triage.no_drift_event")
        return {"severity": "LOW", "reasoning": "No drift event provided"}

    # Simple heuristic: map overall_severity to investigation severity
    drift_report = drift_event_dict.get("drift_report", {})
    overall_severity = drift_report.get("overall_severity", "green")

    severity_mapping = {
        "green": "LOW",
        "yellow": "MEDIUM",
        "red": "HIGH",
    }
    severity = severity_mapping.get(overall_severity, "LOW")

    reasoning = f"Drift report shows {overall_severity} severity"

    log.info(
        "triage.complete",
        investigation_id=state.get("investigation_id"),
        severity=severity,
    )

    return {
        "severity": severity,
        "reasoning": reasoning,
    }
