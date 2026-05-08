"""Action node: construct and enqueue QueuedAction for MEDIUM/HIGH severity."""

import json
import os

import structlog
from agent.app.core.config import AgentSettings
from agent.app.llm.client import get_action_model
from agent.app.schemas.state import AgentState
from shared.contracts import ActionType

log = structlog.get_logger()

ACTION_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "action.txt")


def load_action_prompt() -> str:
    """Load the action prompt from file."""
    with open(ACTION_PROMPT_PATH) as f:
        return f.read()


def generate_idempotency_key(investigation_id: str, action_type: str, target_version: str) -> str:
    """
    Generate deterministic idempotency key.

    Format: {investigation_id}:{action_type}:{target_version}
    Hashed to fixed-length string for database storage.
    """
    import hashlib

    combined = f"{investigation_id}:{action_type}:{target_version}"
    return hashlib.sha256(combined.encode()).hexdigest()


def _get_deterministic_action(severity: str) -> tuple[ActionType, str]:
    """Get deterministic action based on severity (fallback logic)."""
    if severity == "HIGH":
        return "retrain", "HIGH severity drift requires retraining (deterministic)"
    elif severity == "MEDIUM":
        return "replay_test", "MEDIUM severity drift requires test replay (deterministic)"
    else:
        raise ValueError(f"Unexpected severity for action: {severity}")


async def action_node(state: AgentState) -> dict:
    """
    Action node: for MEDIUM/HIGH severity, recommend an action with LLM or deterministic logic.

    With LLM enabled, calls Claude Sonnet with safety constraints:
    - LOW: no action
    - MEDIUM: only replay_test allowed (blocks retrain/rollback)
    - HIGH: retrain and rollback both allowed

    Falls back to deterministic on any error or if LLM disabled.

    Args:
        state: Graph state with severity, investigation_id, model_version, and settings

    Returns:
        Dict with action_queued, status, action_type, idempotency_key (if queued)
    """
    log.info("action.execute", investigation_id=state.get("investigation_id"))

    severity = state.get("severity", "LOW")

    # LOW severity always gets no action, regardless of LLM
    if severity == "LOW":
        log.info("action.skipped_low_severity", investigation_id=state.get("investigation_id"))
        return {"action_queued": False, "status": "skipped", "action_type": None}

    investigation_id = state.get("investigation_id")
    model_version = state.get("model_version")

    # Get settings from state; create from env if not provided (for Studio)
    settings = state.get("_settings") or AgentSettings()

    # If LLM disabled, use deterministic path
    if not settings.use_llm:
        action_type, reasoning = _get_deterministic_action(severity)
        idempotency_key = generate_idempotency_key(investigation_id, action_type, model_version)
        log.info(
            "action.execute.success",
            investigation_id=investigation_id,
            action_type=action_type,
            idempotency_key=idempotency_key,
            source="deterministic",
        )
        return {
            "action_queued": True,
            "status": "paused",
            "action_type": action_type,
            "idempotency_key": idempotency_key,
        }

    # LLM path with safety constraints and deterministic fallback
    try:
        prompt = load_action_prompt()
        model = get_action_model(settings)

        # Format state for LLM
        drift_event = state.get("drift_event", {})
        drift_report = drift_event.get("drift_report", {})

        context = f"""
Investigation ID: {investigation_id}
Model Version: {model_version}
Severity: {severity}
Drift Report: {drift_report}
"""

        user_message = f"{prompt}\n\n{context}"

        # Call LLM with timeout handling
        response = await model.ainvoke({"messages": [("user", user_message)]})

        # Parse LLM response (expecting JSON)
        response_text = response.content if hasattr(response, "content") else str(response)

        # Try to extract JSON from response
        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "{" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
            else:
                json_str = response_text

            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            log.warning(
                "action.llm_json_parse_error",
                investigation_id=investigation_id,
                error=str(e),
            )
            action_type, _ = _get_deterministic_action(severity)
            idempotency_key = generate_idempotency_key(investigation_id, action_type, model_version)
            return {
                "action_queued": True,
                "status": "paused",
                "action_type": action_type,
                "idempotency_key": idempotency_key,
            }

        # Validate action_type value
        recommended_action = result.get("action_type", "").lower()
        if recommended_action not in ("replay_test", "retrain", "rollback"):
            log.warning(
                "action.llm_invalid_action",
                investigation_id=investigation_id,
                action_type=recommended_action,
            )
            action_type, _ = _get_deterministic_action(severity)
            idempotency_key = generate_idempotency_key(investigation_id, action_type, model_version)
            return {
                "action_queued": True,
                "status": "paused",
                "action_type": action_type,
                "idempotency_key": idempotency_key,
            }

        # Apply safety constraints based on severity
        if severity == "MEDIUM":
            # MEDIUM: only allow replay_test
            if recommended_action != "replay_test":
                log.warning(
                    "action.llm_constraint_violation",
                    investigation_id=investigation_id,
                    severity=severity,
                    recommended=recommended_action,
                    allowed="replay_test",
                )
                action_type = "replay_test"
            else:
                action_type: ActionType = "replay_test"  # type: ignore
        elif severity == "HIGH":
            # HIGH: allow retrain and rollback
            if recommended_action not in ("retrain", "rollback"):
                log.warning(
                    "action.llm_constraint_violation_high",
                    investigation_id=investigation_id,
                    recommended=recommended_action,
                    allowed=["retrain", "rollback"],
                )
                action_type = "retrain"
            else:
                action_type = recommended_action  # type: ignore

        # Special handling for rollback: log that it requires HIL approval
        if action_type == "rollback":
            log.info(
                "action.rollback_recommended",
                investigation_id=investigation_id,
                note="Rollback requires HIL approval and will be gated by platform promotion validation",
            )

        idempotency_key = generate_idempotency_key(investigation_id, action_type, model_version)

        log.info(
            "action.execute.success",
            investigation_id=investigation_id,
            action_type=action_type,
            idempotency_key=idempotency_key,
            source="llm",
        )

        return {
            "action_queued": True,
            "status": "paused",
            "action_type": action_type,
            "idempotency_key": idempotency_key,
        }

    except Exception as e:
        log.warning(
            "action.llm_error",
            investigation_id=investigation_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        # Fallback to deterministic on ANY error
        action_type, _ = _get_deterministic_action(severity)
        idempotency_key = generate_idempotency_key(investigation_id, action_type, model_version)
        return {
            "action_queued": True,
            "status": "paused",
            "action_type": action_type,
            "idempotency_key": idempotency_key,
        }
