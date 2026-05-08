"""Comms node: write human-readable summary for HIL inbox."""

import os
from typing import Any

import structlog

log = structlog.get_logger()

COMMS_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "comms.txt")


def load_comms_prompt() -> str:
    """Load the comms prompt from file."""
    with open(COMMS_PROMPT_PATH) as f:
        return f.read()


def comms_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Comms node: write human-readable summary of investigation for HIL inbox.

    For now, generate a simple text summary. LLM integration will be done separately.
    """
    log.info("comms.execute", investigation_id=state.get("investigation_id"))

    investigation_summary = {
        "investigation_id": state.get("investigation_id"),
        "model_name": state.get("model_name"),
        "model_version": state.get("model_version"),
        "severity": state.get("severity"),
        "reasoning": state.get("reasoning", ""),
        "action_type": state.get("action_type", "none"),
    }

    # Generate a simple summary
    summary_text = (
        f"Investigation {investigation_summary['investigation_id']} for {investigation_summary['model_name']} "
        f"(v{investigation_summary['model_version']}) detected {investigation_summary['severity']} severity drift. "
        f"{investigation_summary['reasoning']}"
    )

    log.info(
        "comms.complete",
        investigation_id=state.get("investigation_id"),
        summary_length=len(summary_text),
    )

    return {"summary": summary_text, "status": "complete"}
