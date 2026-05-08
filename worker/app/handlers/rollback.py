"""``rollback`` handler — blocked-DLQ until HIL + promotion gate exist.

Raises ``RollbackBlockedError`` so the loop routes the message to the
DLQ with reason ``rollback_blocked_no_promotion_gate``. A silent success
here would be the worst possible outcome — see plan §8.
"""

from __future__ import annotations

import structlog
from shared.contracts import QueuedAction

from app.handlers.base import HandlerResult, RollbackBlockedError

log = structlog.get_logger(__name__)


async def handle(action: QueuedAction) -> HandlerResult:
    log.error(
        "handler.rollback.blocked",
        idempotency_key=action.idempotency_key,
        investigation_id=action.investigation_id,
        model_name=action.model_name,
        target_version=action.target_version,
    )
    raise RollbackBlockedError(
        f"rollback for {action.model_name}@{action.target_version} blocked: "
        "HIL + promotion gate not yet implemented"
    )
