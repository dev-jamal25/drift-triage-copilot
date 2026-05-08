"""Agent LangGraph state schema."""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """
    Mutable state passed through the LangGraph investigation workflow.

    All fields are optional (total=False) to allow nodes to incrementally build state.
    """

    # Initial state from webhook
    investigation_id: str
    model_name: str
    model_version: str
    drift_event: dict[str, Any]

    # Internal settings (for node access)
    _settings: Any  # AgentSettings instance

    # Supervisor routing
    next: str

    # Triage output
    severity: str
    reasoning: str

    # Action output
    action_type: str
    idempotency_key: str
    action_queued: bool

    # Comms output
    summary: str
    status: str
