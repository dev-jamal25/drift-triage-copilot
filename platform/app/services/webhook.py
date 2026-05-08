"""Drift-event webhook emitter.

POSTs a ``shared.contracts.DriftEvent`` to ``settings.agent_webhook_url`` when
the drift scheduler observes a severity change. Tenacity retries on transport
errors and 5xx responses; failure after the configured attempts logs a warning
and returns ``False`` — webhook failure must never raise into the scheduler
loop, otherwise one bad agent can stall future ticks.
"""

from __future__ import annotations

import httpx
import structlog
from shared.contracts import DriftEvent
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import Settings

log = structlog.get_logger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    """Retry on transient transport issues and on 5xx server errors."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    return False


async def emit_drift_event(
    event: DriftEvent,
    settings: Settings,
    client: httpx.AsyncClient,
) -> bool:
    """POST a DriftEvent to the agent's webhook URL.

    Returns ``True`` on a successful 2xx, ``False`` if no URL is configured
    or the retries are exhausted. Never raises.
    """
    url = settings.agent_webhook_url
    if url is None:
        log.debug("webhook_skipped", reason="no agent_webhook_url configured")
        return False

    payload = event.model_dump(mode="json")
    attempts = max(1, settings.agent_webhook_max_retries)

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=0.5, max=5.0),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        ):
            with attempt:
                response = await client.post(url, json=payload)
                response.raise_for_status()
    except RetryError as exc:
        log.warning(
            "webhook_emit_failed",
            url=url,
            event_id=event.event_id,
            attempts=attempts,
            reason=str(exc),
        )
        return False
    except httpx.HTTPStatusError as exc:
        # Non-retryable 4xx — agent rejected our payload. Don't pretend it worked.
        log.warning(
            "webhook_emit_rejected",
            url=url,
            event_id=event.event_id,
            status_code=exc.response.status_code,
        )
        return False
    except httpx.HTTPError as exc:
        log.warning(
            "webhook_emit_failed",
            url=url,
            event_id=event.event_id,
            attempts=attempts,
            reason=str(exc),
        )
        return False

    log.info(
        "webhook_emit_ok",
        url=url,
        event_id=event.event_id,
        severity=event.severity,
        previous_severity=event.previous_severity,
    )
    return True
