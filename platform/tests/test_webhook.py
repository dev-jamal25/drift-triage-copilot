"""Tests for ``app.services.webhook.emit_drift_event``.

Uses ``httpx.MockTransport`` so we don't open a real socket. Failure modes
must never raise — the drift scheduler relies on this.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from shared.contracts import DriftEvent, DriftReport

from app.core.config import Settings
from app.services.webhook import emit_drift_event


def _settings(url: str | None = "http://agent.local/webhooks/drift", retries: int = 3) -> Settings:
    return Settings(
        agent_webhook_url=url,
        agent_webhook_timeout_s=1.0,
        agent_webhook_max_retries=retries,
        load_model_on_startup=False,
    )


def _event(severity: str = "yellow", previous: str = "green") -> DriftEvent:
    now = datetime.now(UTC)
    return DriftEvent(
        event_id="11111111-1111-1111-1111-111111111111",
        timestamp=now,
        model_name="bank-marketing-classifier",
        model_version="3",
        severity=severity,  # type: ignore[arg-type]
        previous_severity=previous,  # type: ignore[arg-type]
        drift_report=DriftReport(
            window_start=now,
            window_end=now,
            sample_size=42,
            overall_severity=severity,  # type: ignore[arg-type]
            feature_drifts=[],
            output_drift_psi=0.12,
        ),
    )


def _make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=1.0)


@pytest.mark.asyncio
async def test_emit_returns_false_when_no_url_configured() -> None:
    settings = _settings(url=None)
    client = _make_client(lambda req: pytest.fail("transport must not be hit"))
    try:
        ok = await emit_drift_event(_event(), settings, client)
    finally:
        await client.aclose()
    assert ok is False


@pytest.mark.asyncio
async def test_emit_succeeds_on_2xx() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = request.read().decode()
        return httpx.Response(200, json={"received": True})

    client = _make_client(handler)
    try:
        ok = await emit_drift_event(_event(), _settings(), client)
    finally:
        await client.aclose()

    assert ok is True
    assert captured["url"] == "http://agent.local/webhooks/drift"
    assert "11111111-1111-1111-1111-111111111111" in captured["json"]


@pytest.mark.asyncio
async def test_emit_retries_on_5xx_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="agent restarting")
        return httpx.Response(200, json={"ok": True})

    client = _make_client(handler)
    try:
        ok = await emit_drift_event(_event(), _settings(retries=3), client)
    finally:
        await client.aclose()

    assert ok is True
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_emit_returns_false_after_exhausting_retries() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="still down")

    client = _make_client(handler)
    try:
        ok = await emit_drift_event(_event(), _settings(retries=3), client)
    finally:
        await client.aclose()

    assert ok is False
    assert calls["n"] == 3  # exhausted exactly the configured attempts


@pytest.mark.asyncio
async def test_emit_returns_false_on_4xx_without_retrying() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad payload")

    client = _make_client(handler)
    try:
        ok = await emit_drift_event(_event(), _settings(retries=3), client)
    finally:
        await client.aclose()

    assert ok is False
    assert calls["n"] == 1  # 4xx is not retryable


@pytest.mark.asyncio
async def test_emit_returns_false_on_transport_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _make_client(handler)
    try:
        ok = await emit_drift_event(_event(), _settings(retries=2), client)
    finally:
        await client.aclose()

    assert ok is False  # never raises into the caller
