"""
Shared Pydantic contracts.

These schemas define the wire formats between services. Any change here is
a breaking change for at least two services. PRs that modify this file
require approval from both authors.

Conventions:
- All datetimes are timezone-aware UTC.
- All IDs are UUID4 strings unless otherwise noted.
- All severity / status fields are Literal types — extending them is a
  breaking change.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Drift report (computed by platform, embedded in DriftEvent)
# =============================================================================


Severity = Literal["green", "yellow", "red"]


class FeatureDrift(BaseModel):
    """Drift signal for a single feature."""

    model_config = ConfigDict(extra="forbid")

    feature_name: str
    feature_type: Literal["numeric", "categorical"]
    metric: Literal["psi", "chi2"]
    value: float
    severity: Severity


class DriftReport(BaseModel):
    """Aggregated drift over a rolling window."""

    model_config = ConfigDict(extra="forbid")

    window_start: datetime
    window_end: datetime
    sample_size: int = Field(..., ge=0)
    overall_severity: Severity
    feature_drifts: list[FeatureDrift]
    output_drift_psi: float = Field(
        ..., description="PSI of predicted positive class distribution"
    )


# =============================================================================
# Drift event (platform -> agent webhook)
# =============================================================================


class DriftEvent(BaseModel):
    """Webhook payload from platform to agent on severity change."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(..., description="UUID4. Idempotency key for the agent.")
    timestamp: datetime
    model_name: str
    model_version: str
    severity: Severity
    previous_severity: Severity
    drift_report: DriftReport


# =============================================================================
# Queued action (agent -> redis -> worker)
# =============================================================================


ActionType = Literal["replay_test", "retrain", "rollback"]


class QueuedAction(BaseModel):
    """
    Action enqueued by the agent for the worker to execute.

    `idempotency_key` is the SHA256 hash of the canonical format:
        SHA256(f"{investigation_id}:{action_type}:{target_version}").hexdigest()
    This ensures idempotent retries — the same action never produces duplicate side effects.
    """

    model_config = ConfigDict(extra="forbid")

    idempotency_key: str
    investigation_id: str
    model_name: str
    action_type: ActionType
    target_version: str
    payload: dict = Field(default_factory=dict)
    attempt: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1)
    created_at: datetime


# =============================================================================
# Promotion (agent -> platform, after HIL approval)
# =============================================================================

PromotionStage = Literal["Staging", "Production", "Archived"]


class PromotionRequest(BaseModel):
    """Request to change a model version's stage in the registry."""

    model_config = ConfigDict(extra="forbid")

    approval_token: str = Field(
        ..., description="Signed token issued by agent on HIL approval."
    )
    investigation_id: str
    target_version: str
    target_stage: PromotionStage
    requested_by: str = Field(..., description="Identifier of the human approver.")


class PromotionResult(BaseModel):
    """Response from platform after evaluating the promotion request."""

    model_config = ConfigDict(extra="forbid")

    accepted: bool
    new_stage: PromotionStage | None = None
    failed_gate: str | None = Field(
        default=None,
        description="Name of the assertion that failed, if accepted=False.",
    )
    message: str


# =============================================================================
# HIL approval (dashboard -> agent, via DB row update)
# =============================================================================


class HilApproval(BaseModel):
    """Pending or resolved approval surfaced in the dashboard."""

    model_config = ConfigDict(extra="forbid")

    approval_id: str
    investigation_id: str
    model_name: str
    proposed_action: ActionType
    target_version: str
    summary: str = Field(..., description="Comms-agent-written human summary.")
    status: Literal["pending", "approved", "denied", "superseded"]
    created_at: datetime
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    superseded_by: str | None = Field(
        default=None,
        description="investigation_id of the newer investigation, if any.",
    )
