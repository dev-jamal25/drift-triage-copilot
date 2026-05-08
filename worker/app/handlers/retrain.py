"""``retrain`` handler — Step 4 safe stub.

Logs and returns ``success``. **Intentionally inert.** The real retrain
path is gated behind two pieces that don't exist yet:
  1. A platform endpoint that accepts an agent ``approval_token`` and
     promotes a candidate model version in MLflow.
  2. The agent's HIL flow surfacing the human approval that mints the
     token.

Until both ship, this stub must NOT touch the model registry or call
back into the platform. No httpx, no mlflow, no shared registry writes.
"""

from __future__ import annotations

import structlog
from shared.contracts import QueuedAction

from app.handlers.base import HandlerResult

log = structlog.get_logger(__name__)


async def handle(action: QueuedAction) -> HandlerResult:
    log.info(
        "handler.retrain.stub",
        idempotency_key=action.idempotency_key,
        investigation_id=action.investigation_id,
        model_name=action.model_name,
        target_version=action.target_version,
    )
    return HandlerResult(status="success")
