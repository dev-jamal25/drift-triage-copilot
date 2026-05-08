"""Comms node: write human-readable summary for HIL inbox."""

import os

import structlog
from agent.app.core.config import AgentSettings
from agent.app.llm.client import get_comms_model
from agent.app.schemas.state import AgentState

log = structlog.get_logger()

COMMS_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "comms.txt")


def load_comms_prompt() -> str:
    """Load the comms prompt from file."""
    with open(COMMS_PROMPT_PATH) as f:
        return f.read()


def _get_deterministic_summary(state: AgentState) -> str:
    """Generate deterministic summary from state (fallback logic)."""
    investigation_summary = {
        "investigation_id": state.get("investigation_id"),
        "model_name": state.get("model_name"),
        "model_version": state.get("model_version"),
        "severity": state.get("severity"),
        "reasoning": state.get("reasoning", ""),
        "action_type": state.get("action_type", "none"),
    }

    summary_text = (
        f"Investigation {investigation_summary['investigation_id']} for {investigation_summary['model_name']} "
        f"(v{investigation_summary['model_version']}) detected {investigation_summary['severity']} severity drift. "
        f"{investigation_summary['reasoning']} Recommended action: {investigation_summary['action_type']}."
    )
    return summary_text


async def comms_node(state: AgentState) -> dict:
    """
    Comms node: generate human-readable summary of investigation for HIL inbox.

    With LLM enabled, calls Claude Haiku to generate a natural summary.
    Falls back to deterministic summary on any error or if LLM disabled.

    Args:
        state: Graph state with investigation details and settings

    Returns:
        Dict with summary (plain text) and status
    """
    log.info("comms.execute", investigation_id=state.get("investigation_id"))

    # Get settings from state; create from env if not provided (for Studio)
    settings = state.get("_settings") or AgentSettings()

    # If LLM disabled, use deterministic path
    if not settings.use_llm:
        summary_text = _get_deterministic_summary(state)
        log.info(
            "comms.complete",
            investigation_id=state.get("investigation_id"),
            summary_length=len(summary_text),
            source="deterministic",
        )
        return {"summary": summary_text, "status": "complete"}

    # LLM path with deterministic fallback
    try:
        prompt = load_comms_prompt()
        model = get_comms_model(settings)

        # Format state for LLM
        context = f"""
Investigation ID: {state.get("investigation_id")}
Model: {state.get("model_name")} (version {state.get("model_version")})
Severity: {state.get("severity")}
Drift Reasoning: {state.get("reasoning", "")}
Recommended Action: {state.get("action_type", "none")}
"""

        user_message = f"{prompt}\n\n{context}"

        # Call LLM with timeout handling
        response = await model.ainvoke({"messages": [("user", user_message)]})

        # Extract plain text response
        summary_text = response.content if hasattr(response, "content") else str(response)
        summary_text = summary_text.strip()

        if not summary_text:
            # Empty response, fall back to deterministic
            log.warning(
                "comms.llm_empty_response",
                investigation_id=state.get("investigation_id"),
            )
            summary_text = _get_deterministic_summary(state)
            return {"summary": summary_text, "status": "complete"}

        log.info(
            "comms.complete",
            investigation_id=state.get("investigation_id"),
            summary_length=len(summary_text),
            source="llm",
        )

        return {"summary": summary_text, "status": "complete"}

    except Exception as e:
        log.warning(
            "comms.llm_error",
            investigation_id=state.get("investigation_id"),
            error=str(e),
            error_type=type(e).__name__,
        )
        # Fallback to deterministic on ANY error
        summary_text = _get_deterministic_summary(state)
        return {"summary": summary_text, "status": "complete"}
