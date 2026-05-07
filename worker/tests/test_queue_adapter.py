"""RedisQueueAdapter unit tests — Step 2.

Pure unit tests using ``unittest.mock.AsyncMock``. No real Redis needed.
The contract these tests pin down:
  - ``claim`` issues a single ``BLMOVE`` from ``:ready`` to ``:processing``.
  - Every transition out of ``:processing`` copies first, then ``LREM``.
  - DLQ envelope contains ``raw``, ``reason``, ``error_type``,
    ``failed_at``, ``attempt`` (correction 10 from the plan).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from app.queue.adapter import RedisQueueAdapter
from app.queue.keys import DLQ, PROCESSING, READY, RETRY


@pytest.fixture
def client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def adapter(client: AsyncMock) -> RedisQueueAdapter:
    return RedisQueueAdapter(client)


def _ordered_method_names(client: AsyncMock) -> list[str]:
    """Names of the sub-method calls on ``client`` in the order they ran."""
    return [c[0] for c in client.mock_calls]


async def test_claim_uses_blmove_ready_to_processing(
    client: AsyncMock, adapter: RedisQueueAdapter
) -> None:
    client.blmove.return_value = b"payload"

    raw = await adapter.claim(timeout_seconds=2.0)

    assert raw == b"payload"
    client.blmove.assert_awaited_once_with(READY, PROCESSING, 2.0, src="RIGHT", dest="LEFT")


async def test_claim_returns_none_on_timeout(client: AsyncMock, adapter: RedisQueueAdapter) -> None:
    client.blmove.return_value = None
    assert await adapter.claim(timeout_seconds=0.1) is None


async def test_ack_lrems_one_from_processing(client: AsyncMock, adapter: RedisQueueAdapter) -> None:
    await adapter.ack(b"payload")
    client.lrem.assert_awaited_once_with(PROCESSING, 1, b"payload")


async def test_nack_retry_zadds_before_lrem(
    client: AsyncMock,
    adapter: RedisQueueAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.queue.adapter._now_epoch", lambda: 1000.0)

    await adapter.nack_retry(b"payload", delay_seconds=5.0)

    assert _ordered_method_names(client) == ["zadd", "lrem"]
    client.zadd.assert_awaited_once_with(RETRY, {b"payload": 1005.0})
    client.lrem.assert_awaited_once_with(PROCESSING, 1, b"payload")


async def test_nack_retry_with_retry_raw_uses_separate_payloads(
    client: AsyncMock,
    adapter: RedisQueueAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``retry_raw`` lets the loop schedule a bumped-attempt copy onto
    ``:retry`` while the *original* in-flight bytes get LREM'd from
    ``:processing``."""
    monkeypatch.setattr("app.queue.adapter._now_epoch", lambda: 1000.0)

    await adapter.nack_retry(
        b"original",
        delay_seconds=5.0,
        retry_raw=b"bumped_attempt_payload",
    )

    assert _ordered_method_names(client) == ["zadd", "lrem"]
    client.zadd.assert_awaited_once_with(RETRY, {b"bumped_attempt_payload": 1005.0})
    client.lrem.assert_awaited_once_with(PROCESSING, 1, b"original")


async def test_dead_letter_lpushes_before_lrem(
    client: AsyncMock,
    adapter: RedisQueueAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.queue.adapter._now_iso", lambda: "2026-05-07T12:00:00+00:00")

    await adapter.dead_letter(
        b'{"hello": "world"}',
        reason="validation_error",
        error_type="ValidationError",
        attempt=0,
    )

    assert _ordered_method_names(client) == ["lpush", "lrem"]


async def test_dead_letter_envelope_shape(
    client: AsyncMock,
    adapter: RedisQueueAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.queue.adapter._now_iso", lambda: "2026-05-07T12:00:00+00:00")

    await adapter.dead_letter(
        b'{"hello": "world"}',
        reason="max_attempts_exhausted",
        error_type="HandlerError",
        attempt=3,
    )

    args, _kwargs = client.lpush.await_args
    queue, envelope_bytes = args
    assert queue == DLQ
    envelope = json.loads(envelope_bytes)
    assert envelope == {
        "raw": '{"hello": "world"}',
        "reason": "max_attempts_exhausted",
        "error_type": "HandlerError",
        "failed_at": "2026-05-07T12:00:00+00:00",
        "attempt": 3,
    }


async def test_dead_letter_lrem_uses_original_raw_not_envelope(
    client: AsyncMock, adapter: RedisQueueAdapter
) -> None:
    """LREM must remove the *original* payload from ``:processing``,
    not the envelope. Otherwise the in-flight message leaks forever."""
    raw = b'{"original": true}'
    await adapter.dead_letter(raw, reason="r", error_type="e", attempt=0)
    client.lrem.assert_awaited_once_with(PROCESSING, 1, raw)


async def test_dead_letter_handles_non_utf8_payload(
    client: AsyncMock, adapter: RedisQueueAdapter
) -> None:
    """Bytes that aren't valid UTF-8 must not crash the DLQ path."""
    bad = b"\xff\xfe garbage"

    await adapter.dead_letter(bad, reason="r", error_type="e", attempt=0)

    args, _ = client.lpush.await_args
    envelope = json.loads(args[1])
    # ``errors='replace'`` substitutes U+FFFD; the contract is just that
    # we get a string and the call did not raise.
    assert isinstance(envelope["raw"], str)
