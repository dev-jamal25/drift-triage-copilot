"""Dispatcher — Step 4.

Maps ``QueuedAction.action_type`` to its handler. Unknown action types
return ``terminal_failure`` so the loop DLQs them; rollback handlers
raise ``RollbackBlockedError`` and the loop catches it at its single
DLQ-routing site.

The "unknown action_type" path is dead today (the ``Literal`` in
``shared.contracts.QueuedAction`` blocks any other value at validation
time). It stays as a defensive guard for the case where
``shared/contracts.py`` adds a new ``action_type`` before the worker
ships its handler.
"""

from __future__ import annotations

import structlog
from shared.contracts import QueuedAction

from app.handlers import replay_test, retrain, rollback
from app.handlers.base import ActionHandler, HandlerResult

log = structlog.get_logger(__name__)

_HANDLERS: dict[str, ActionHandler] = {
    "replay_test": replay_test.handle,
    "retrain": retrain.handle,
    "rollback": rollback.handle,
}


async def dispatch(action: QueuedAction) -> HandlerResult:
    handler = _HANDLERS.get(action.action_type)
    if handler is None:
        log.error(
            "dispatcher.unknown_action_type",
            action_type=action.action_type,
            idempotency_key=action.idempotency_key,
        )
        return HandlerResult(
            status="terminal_failure",
            error_type="UnknownActionType",
            error_msg=f"no handler registered for action_type={action.action_type!r}",
        )
    return await handler(action)
