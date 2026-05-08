"""Handler protocol, result type, and the rollback sentinel exception.

A handler is an ``async def handle(action: QueuedAction) -> HandlerResult``
coroutine. The dispatcher routes by ``action.action_type``; the loop
reads the returned ``HandlerResult`` to decide ack / retry / DLQ.

``RollbackBlockedError`` is raised (not returned) by the rollback
handler so the loop has a clear place to catch and DLQ with the
plan-specified reason. Step 5 will introduce retry routing for
``retryable_failure``; Step 4 only produces ``success`` and
``terminal_failure``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from shared.contracts import QueuedAction

HandlerStatus = Literal["success", "retryable_failure", "terminal_failure"]


@dataclass(frozen=True)
class HandlerResult:
    """Outcome of a single dispatch.

    ``error_type`` is the exception class name or a stable tag like
    ``"UnknownActionType"``. ``error_msg`` is human-readable detail for
    logs (not embedded in the DLQ envelope).
    """

    status: HandlerStatus
    error_type: str | None = None
    error_msg: str | None = None


ActionHandler = Callable[[QueuedAction], Awaitable[HandlerResult]]


class RollbackBlockedError(Exception):
    """Raised by the rollback handler until HIL + promotion gate exist.

    The loop catches this and DLQs with reason
    ``rollback_blocked_no_promotion_gate``. Per the plan: rollback must
    fail loudly, never silently succeed.
    """
