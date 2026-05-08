"""Triage node: classify drift severity from DriftEvent."""

import json
import os
from typing import Any

import structlog
from agent.app.core.config import AgentSettings
from agent.app.llm.client import get_triage_model
from agent.app.schemas.state import AgentState

log = structlog.get_logger()

TRIAGE_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "triage.txt")


def load_triage_prompt() -> str:
    """Load the triage prompt from file."""
    with open(TRIAGE_PROMPT_PATH) as f:
        return f.read()


def _get_deterministic_severity(drift_event_dict: dict[str, Any]) -> tuple[str, str]:
    """Classify severity based on drift_event's overall_severity (deterministic fallback)."""
    drift_report = drift_event_dict.get("drift_report", {})
    overall_severity = drift_report.get("overall_severity", "green")

    severity_mapping = {
        "green": "LOW",
        "yellow": "MEDIUM",
        "red": "HIGH",
    }
    severity = severity_mapping.get(overall_severity, "LOW")
    reasoning = f"Drift report shows {overall_severity} severity (deterministic)"
    return severity, reasoning


async def triage_node(state: AgentState) -> dict:
    """
    Triage node: classify drift severity from DriftEvent into LOW / MEDIUM / HIGH.

    With LLM enabled (AGENT_USE_LLM=true), calls Claude Haiku for classification.
    Falls back to deterministic mapping on any error or if LLM disabled.

    Args:
        state: Graph state with drift_event, investigation_id, and settings

    Returns:
        Dict with severity (LOW|MEDIUM|HIGH) and reasoning
    """
    log.info("triage.execute", investigation_id=state.get("investigation_id"))

    drift_event_dict = state.get("drift_event")
    if not drift_event_dict:
        log.error("triage.no_drift_event")
        return {"severity": "LOW", "reasoning": "No drift event provided"}

    # Get settings from state; create from env if not provided (for Studio)
    settings = state.get("_settings") or AgentSettings()

    # If LLM disabled, use deterministic path
    if not settings.use_llm:
        severity, reasoning = _get_deterministic_severity(drift_event_dict)
        log.info(
            "triage.complete",
            investigation_id=state.get("investigation_id"),
            severity=severity,
            source="deterministic",
        )
        return {"severity": severity, "reasoning": reasoning}

    # LLM path with deterministic fallback
    try:
        prompt = load_triage_prompt()
        model = get_triage_model(settings)

        # Format drift event for LLM
        drift_report = drift_event_dict.get("drift_report", {})
        model_name = drift_event_dict.get("model_name", "unknown")
        model_version = drift_event_dict.get("model_version", "unknown")

        context = f"""
Model: {model_name} (version {model_version})
Overall Severity: {drift_report.get("overall_severity", "unknown")}
Numeric Drift (PSI): {drift_report.get("numeric_drift", {})}
Categorical Drift (Chi²): {drift_report.get("categorical_drift", {})}
Drift Metrics: {drift_report.get("metrics", {})}
"""

        user_message = f"{prompt}\n\n{context}"

        # Call LLM with timeout handling
        response = await model.ainvoke({"messages": [("user", user_message)]})

        # Parse LLM response (expecting JSON)
        response_text = response.content if hasattr(response, "content") else str(response)

        # Try to extract JSON from response
        try:
            # Look for JSON block in response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "{" in response_text:
                # Try to find JSON object
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
            else:
                json_str = response_text

            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            log.warning(
                "triage.llm_json_parse_error",
                investigation_id=state.get("investigation_id"),
                error=str(e),
            )
            severity, reasoning = _get_deterministic_severity(drift_event_dict)
            return {
                "severity": severity,
                "reasoning": reasoning + " (LLM JSON parse failed, fell back to deterministic)",
            }

        # Validate severity value
        severity = result.get("severity", "").upper()
        if severity not in ("LOW", "MEDIUM", "HIGH"):
            log.warning(
                "triage.llm_invalid_severity",
                investigation_id=state.get("investigation_id"),
                severity=severity,
            )
            severity, reasoning = _get_deterministic_severity(drift_event_dict)
            return {
                "severity": severity,
                "reasoning": reasoning
                + " (LLM returned invalid severity, fell back to deterministic)",
            }

        reasoning = result.get("reasoning", "LLM classification")

        log.info(
            "triage.complete",
            investigation_id=state.get("investigation_id"),
            severity=severity,
            source="llm",
        )

        return {"severity": severity, "reasoning": reasoning}

    except Exception as e:
        log.warning(
            "triage.llm_error",
            investigation_id=state.get("investigation_id"),
            error=str(e),
            error_type=type(e).__name__,
        )
        # Fallback to deterministic on ANY error
        severity, reasoning = _get_deterministic_severity(drift_event_dict)
        return {
            "severity": severity,
            "reasoning": reasoning
            + f" (LLM error: {type(e).__name__}, fell back to deterministic)",
        }
