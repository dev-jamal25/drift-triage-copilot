"""``replay_test`` handler — Step 4 safe stub.

Logs and returns ``success``. The real replay path (re-scoring a frozen
batch against a candidate model) lands once the platform exposes a
batch-prediction endpoint we can call back into.
"""

from __future__ import annotations

import structlog
from shared.contracts import QueuedAction

from app.handlers.base import HandlerResult

log = structlog.get_logger(__name__)


async def handle(action: QueuedAction) -> HandlerResult:
    log.info(
        "handler.replay_test.stub",
        idempotency_key=action.idempotency_key,
        investigation_id=action.investigation_id,
        model_name=action.model_name,
        target_version=action.target_version,
    )
    return HandlerResult(status="success")
