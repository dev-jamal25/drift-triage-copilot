"""Dispatcher routing tests — Step 4."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import pytest
from shared.contracts import ActionType, QueuedAction

from app.handlers.base import RollbackBlockedError
from app.handlers.dispatcher import dispatch


def _action(action_type: str) -> QueuedAction:
    return QueuedAction(
        idempotency_key=f"inv1:{action_type}:v3",
        investigation_id="inv1",
        model_name="bank-marketing-classifier",
        # cast bypasses the Literal at construction time only; QueuedAction
        # would normally reject anything outside the literal at validation.
        action_type=cast(ActionType, action_type),
        target_version="v3",
        payload={},
        attempt=0,
        max_attempts=3,
        created_at=datetime(2026, 5, 7, tzinfo=UTC),
    )


async def test_replay_test_routes_to_replay_handler() -> None:
    result = await dispatch(_action("replay_test"))
    assert result.status == "success"


async def test_retrain_routes_to_retrain_handler() -> None:
    result = await dispatch(_action("retrain"))
    assert result.status == "success"


async def test_rollback_raises_rollback_blocked_error() -> None:
    """The dispatcher does NOT catch RollbackBlockedError; the loop does.
    Keeping the catch out of the dispatcher preserves a single failure-
    routing site in the loop."""
    with pytest.raises(RollbackBlockedError):
        await dispatch(_action("rollback"))


async def test_unknown_action_returns_terminal_failure() -> None:
    """Defensive guard for the case where ``shared.contracts`` adds a new
    ``action_type`` before the worker ships its handler. The Literal in
    ``QueuedAction.action_type`` blocks this in production today, so we
    mutate the field after construction to force the path."""
    action = _action("replay_test")
    object.__setattr__(action, "action_type", "future_unknown_type")

    result = await dispatch(action)

    assert result.status == "terminal_failure"
    assert result.error_type == "UnknownActionType"
    assert result.error_msg is not None
    assert "future_unknown_type" in result.error_msg
