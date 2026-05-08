"""Safe-stub tests — Step 4.

Pin down what the stubs may and may NOT do. Most importantly: ``retrain``
must not import anything that could mutate the model registry or call
back into the platform's promotion endpoint.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest
from shared.contracts import QueuedAction

from app.handlers import replay_test, retrain, rollback
from app.handlers.base import RollbackBlockedError


def _action(action_type: str) -> QueuedAction:
    return QueuedAction(
        idempotency_key=f"inv1:{action_type}:v3",
        investigation_id="inv1",
        model_name="bank-marketing-classifier",
        action_type=action_type,  # type: ignore[arg-type]
        target_version="v3",
        payload={},
        attempt=0,
        max_attempts=3,
        created_at=datetime(2026, 5, 7, tzinfo=UTC),
    )


async def test_replay_test_returns_success() -> None:
    result = await replay_test.handle(_action("replay_test"))
    assert result.status == "success"
    assert result.error_type is None
    assert result.error_msg is None


async def test_retrain_returns_success() -> None:
    result = await retrain.handle(_action("retrain"))
    assert result.status == "success"
    assert result.error_type is None
    assert result.error_msg is None


async def test_rollback_raises_rollback_blocked_error() -> None:
    with pytest.raises(RollbackBlockedError) as excinfo:
        await rollback.handle(_action("rollback"))
    # The message must mention what was being rolled back so DLQ readers
    # can tell which model/version was blocked.
    assert "bank-marketing-classifier" in str(excinfo.value)
    assert "v3" in str(excinfo.value)


def _import_lines(module: object) -> str:
    """Return only the ``import`` and ``from ... import`` lines of a module's
    source. Avoids false positives from docstrings or comments."""
    src = inspect.getsource(module)
    return "\n".join(
        line for line in src.splitlines() if line.startswith("import ") or line.startswith("from ")
    )


def test_retrain_module_does_not_import_promotion_paths() -> None:
    """Belt-and-suspenders: retrain must NOT import anything that could
    mutate the registry or call back into the platform. If you're adding
    real retrain logic, that code belongs behind the HIL + promotion
    gate, not in this stub."""
    imports = _import_lines(retrain)
    forbidden = ("mlflow", "httpx", "requests", "promote")
    for needle in forbidden:
        assert (
            needle not in imports
        ), f"retrain stub imports {needle!r}; that lands behind HIL + /promote"


def test_rollback_module_does_not_import_promotion_paths() -> None:
    """Same guard for rollback — it's the riskiest action and must stay
    inert until the promotion gate exists."""
    imports = _import_lines(rollback)
    forbidden = ("mlflow", "httpx", "requests", "promote")
    for needle in forbidden:
        assert (
            needle not in imports
        ), f"rollback stub imports {needle!r}; that lands behind HIL + /promote"
