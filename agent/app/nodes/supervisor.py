"""Supervisor node: entry point, routing, and stale URI check."""

from typing import Any

import structlog

log = structlog.get_logger()


def supervisor_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Supervisor node: routes between sub-agents.

    On resume, checks if model URI is still valid (stub for now).
    """
    log.info("supervisor.execute", investigation_id=state.get("investigation_id"))

    # Route based on the current state
    # Initial path: triage -> action (if needed) -> comms
    if "severity" not in state:
        return {"next": "triage"}

    severity = state.get("severity")
    if severity in ("MEDIUM", "HIGH"):
        if "action_queued" not in state:
            return {"next": "action"}

    return {"next": "comms"}
