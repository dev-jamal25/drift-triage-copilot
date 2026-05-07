"""Tests for ``app.services.drift_scheduler.DriftScheduler``.

The scheduler's job is twofold:
  * drive ``recompute_drift`` on a cadence,
  * cache ``latest_report`` + ``previous_severity`` for ``GET /drift`` and
    Step 5's webhook to read.

Tests mock both ``recompute_drift`` and the session factory so the scheduler
runs purely in-process — no DB, no MLflow.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from shared.contracts import DriftReport

from app.core.config import Settings
from app.services.drift_scheduler import DriftScheduler
from app.services.model_loader import ModelBundle


def _settings(**overrides) -> Settings:
    base = {
        "drift_recompute_interval_s": 0.05,
        "drift_window_size": 100,
        "load_model_on_startup": False,
    }
    base.update(overrides)
    return Settings(**base)


def _bundle(*, stats: dict | None) -> ModelBundle:
    return ModelBundle(
        pipeline=MagicMock(),
        threshold=0.4,
        model_uri="models:/m@staging",
        model_name="m",
        version="1",
        alias="staging",
        run_id="r1",
        loaded_at=datetime.now(UTC),
        reference_stats=stats,
    )


def _factory_returning(_session: AsyncMock):
    """Mimic an ``async_sessionmaker``: callable returning an async ctx manager."""

    @asynccontextmanager
    async def _ctx():
        yield _session

    def _call():
        return _ctx()

    return _call


def _report(severity: str, n: int = 5) -> DriftReport:
    now = datetime.now(UTC)
    return DriftReport(
        window_start=now,
        window_end=now,
        sample_size=n,
        overall_severity=severity,  # type: ignore[arg-type]
        feature_drifts=[],
        output_drift_psi=0.0,
    )


@pytest.mark.asyncio
async def test_tick_skips_when_no_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    called = AsyncMock()
    monkeypatch.setattr("app.services.drift_scheduler.recompute_drift", called)
    sch = DriftScheduler(
        settings=_settings(),
        session_factory=_factory_returning(AsyncMock()),
        get_bundle=lambda: None,
    )
    await sch.tick()
    called.assert_not_awaited()
    assert sch.latest_report is None


@pytest.mark.asyncio
async def test_tick_skips_when_reference_stats_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = AsyncMock()
    monkeypatch.setattr("app.services.drift_scheduler.recompute_drift", called)
    sch = DriftScheduler(
        settings=_settings(),
        session_factory=_factory_returning(AsyncMock()),
        get_bundle=lambda: _bundle(stats=None),
    )
    await sch.tick()
    called.assert_not_awaited()
    assert sch.latest_report is None


@pytest.mark.asyncio
async def test_tick_populates_latest_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.services.drift_scheduler.recompute_drift",
        AsyncMock(return_value=_report("green", n=10)),
    )
    sch = DriftScheduler(
        settings=_settings(),
        session_factory=_factory_returning(AsyncMock()),
        get_bundle=lambda: _bundle(stats={"numeric": {}, "categorical": {}, "output": {}}),
    )
    await sch.tick()
    assert sch.latest_report is not None
    assert sch.latest_report.sample_size == 10
    assert sch.last_refreshed_at is not None
    assert sch.previous_severity is None  # first tick has no predecessor


@pytest.mark.asyncio
async def test_tick_tracks_previous_severity_across_ticks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    severities = iter([_report("green"), _report("yellow"), _report("red")])
    monkeypatch.setattr(
        "app.services.drift_scheduler.recompute_drift",
        AsyncMock(side_effect=lambda *a, **kw: next(severities)),
    )
    sch = DriftScheduler(
        settings=_settings(),
        session_factory=_factory_returning(AsyncMock()),
        get_bundle=lambda: _bundle(stats={"numeric": {}, "categorical": {}, "output": {}}),
    )

    await sch.tick()
    assert sch.previous_severity is None
    assert sch.latest_report.overall_severity == "green"

    await sch.tick()
    assert sch.previous_severity == "green"
    assert sch.latest_report.overall_severity == "yellow"

    await sch.tick()
    assert sch.previous_severity == "yellow"
    assert sch.latest_report.overall_severity == "red"


@pytest.mark.asyncio
async def test_start_then_stop_is_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.services.drift_scheduler.recompute_drift",
        AsyncMock(return_value=_report("green")),
    )
    sch = DriftScheduler(
        settings=_settings(drift_recompute_interval_s=10.0),  # long sleep, we'll cancel
        session_factory=_factory_returning(AsyncMock()),
        get_bundle=lambda: _bundle(stats={"numeric": {}, "categorical": {}, "output": {}}),
    )
    sch.start()
    # second start is a no-op
    sch.start()
    assert sch._task is not None
    await sch.stop()
    assert sch._task is None
    # double-stop is also fine
    await sch.stop()
