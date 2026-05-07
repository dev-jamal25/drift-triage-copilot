"""Background asyncio task that recomputes drift on a fixed cadence.

Started in the FastAPI lifespan, cancelled cleanly on shutdown. The scheduler
caches ``latest_report`` for ``GET /drift`` to read and tracks
``previous_severity`` so Step 5 can detect severity changes and emit
DriftEvent webhooks.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from datetime import UTC, datetime

import structlog
from shared.contracts import DriftReport, Severity
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import Settings
from app.services.drift_service import recompute_drift
from app.services.model_loader import ModelBundle

log = structlog.get_logger(__name__)


class DriftScheduler:
    """Owns the recompute loop and the latest-report cache."""

    def __init__(
        self,
        *,
        settings: Settings,
        session_factory: async_sessionmaker,
        get_bundle: Callable[[], ModelBundle | None],
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._get_bundle = get_bundle
        self._task: asyncio.Task | None = None
        self.latest_report: DriftReport | None = None
        self.previous_severity: Severity | None = None
        self.last_refreshed_at: datetime | None = None

    def start(self) -> None:
        """Launch the recompute loop. Idempotent."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop(), name="drift-scheduler")
        log.info(
            "drift_scheduler_started",
            interval_s=self._settings.drift_recompute_interval_s,
            window_size=self._settings.drift_window_size,
        )

    async def stop(self) -> None:
        """Cancel and await the recompute loop. Idempotent."""
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        log.info("drift_scheduler_stopped")

    async def tick(self) -> None:
        """Recompute drift once. Public so tests can drive it deterministically."""
        bundle = self._get_bundle()
        if bundle is None:
            log.warning("drift_tick_skipped", reason="no model bundle loaded")
            return
        if bundle.reference_stats is None:
            log.warning(
                "drift_tick_skipped",
                reason="bundle has no reference_stats — re-register the model with Step 2",
                model_version=bundle.version,
            )
            return

        async with self._session_factory() as session:
            report = await recompute_drift(session, bundle, self._settings.drift_window_size)

        self.previous_severity = (
            self.latest_report.overall_severity if self.latest_report is not None else None
        )
        self.latest_report = report
        self.last_refreshed_at = datetime.now(UTC)

    async def _loop(self) -> None:
        """Run ``tick`` forever. Catches anything but ``CancelledError`` so a
        single failed tick doesn't kill the loop."""
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except (RuntimeError, ValueError, OSError):
                log.exception("drift_tick_failed")
            await asyncio.sleep(self._settings.drift_recompute_interval_s)
