"""Action node: construct and enqueue QueuedAction for MEDIUM/HIGH severity."""

import hashlib
from typing import Any

import structlog
from shared.contracts import ActionType

log = structlog.get_logger()


def generate_idempotency_key(investigation_id: str, action_type: str, target_version: str) -> str:
    """
    Generate deterministic idempotency key.

    Format: {investigation_id}:{action_type}:{target_version}
    Hashed to fixed-length string for database storage.
    """
    combined = f"{investigation_id}:{action_type}:{target_version}"
    return hashlib.sha256(combined.encode()).hexdigest()


def action_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Action node: for MEDIUM/HIGH severity, construct and return QueuedAction.

    The actual enqueue and database write happens in the router after graph execution.
    """
    log.info("action.execute", investigation_id=state.get("investigation_id"))

    severity = state.get("severity", "LOW")
    if severity not in ("MEDIUM", "HIGH"):
        log.info("action.skipped", severity=severity)
        return {"action_queued": False, "status": "skipped"}

    investigation_id = state.get("investigation_id")
    model_version = state.get("model_version")

    # Determine recommended action based on severity
    action_type: ActionType = "retrain" if severity == "HIGH" else "replay_test"

    # Generate idempotency key
    idempotency_key = generate_idempotency_key(investigation_id, action_type, model_version)

    log.info(
        "action.execute.success",
        investigation_id=investigation_id,
        action_type=action_type,
        idempotency_key=idempotency_key,
    )

    return {
        "action_queued": True,
        "status": "paused",
        "action_type": action_type,
        "idempotency_key": idempotency_key,
    }
