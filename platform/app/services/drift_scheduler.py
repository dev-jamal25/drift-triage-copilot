"""Background asyncio task that recomputes drift on a fixed cadence.

Started in the FastAPI lifespan, cancelled cleanly on shutdown. The scheduler
caches ``latest_report`` for ``GET /drift`` to read and tracks
``previous_severity`` so Step 5 can detect severity changes and emit
DriftEvent webhooks.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import structlog
from shared.contracts import DriftEvent, DriftReport, Severity
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import Settings
from app.services.drift_service import recompute_drift
from app.services.model_loader import ModelBundle
from app.services.webhook import emit_drift_event

log = structlog.get_logger(__name__)


class DriftScheduler:
    """Owns the recompute loop, the latest-report cache, and severity-change
    webhook emission."""

    def __init__(
        self,
        *,
        settings: Settings,
        session_factory: async_sessionmaker,
        get_bundle: Callable[[], ModelBundle | None],
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._get_bundle = get_bundle
        self._http_client = http_client
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

        # Capture pre-update state for transition detection.
        old_latest = self.latest_report

        self.previous_severity = old_latest.overall_severity if old_latest is not None else None
        self.latest_report = report
        self.last_refreshed_at = datetime.now(UTC)

        await self._maybe_emit_webhook(bundle, old_latest, report)

    async def _maybe_emit_webhook(
        self,
        bundle: ModelBundle,
        old_latest: DriftReport | None,
        report: DriftReport,
    ) -> None:
        """If severity changed, fire ``DriftEvent`` to the agent webhook.

        First-tick semantics: a service that comes up in non-green should
        still alert the agent. We treat "no prior tick" as ``green`` so the
        first transition green→{yellow,red} is a real change.
        """
        if self._http_client is None or self._settings.agent_webhook_url is None:
            return

        effective_previous: Severity = (
            old_latest.overall_severity if old_latest is not None else "green"
        )
        if report.overall_severity == effective_previous:
            return

        assert self.last_refreshed_at is not None  # we just set it
        event = DriftEvent(
            event_id=str(uuid.uuid4()),
            timestamp=self.last_refreshed_at,
            model_name=bundle.model_name,
            model_version=bundle.version,
            severity=report.overall_severity,
            previous_severity=effective_previous,
            drift_report=report,
        )
        log.info(
            "drift_severity_change",
            previous=effective_previous,
            current=report.overall_severity,
            event_id=event.event_id,
        )
        await emit_drift_event(event, self._settings, self._http_client)

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
