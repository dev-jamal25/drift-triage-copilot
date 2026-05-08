"""WorkerSettings env parsing — Step 1.

Pure unit tests; no Redis, no Postgres. Each test isolates the env so the
host's real ``.env`` cannot leak into the result.
"""

from __future__ import annotations

import pytest

from app.core.config import WorkerSettings, redact_url

_ALL_VARS = (
    "LOG_LEVEL",
    "WORKER_LOG_LEVEL",
    "DATABASE_URL",
    "WORKER_DATABASE_URL",
    "REDIS_URL",
    "WORKER_REDIS_URL",
    "MODEL_SERVICE_URL",
    "WORKER_MODEL_SERVICE_URL",
    "WORKER_MAX_RETRIES",
    "WORKER_BACKOFF_BASE_SECONDS",
    "WORKER_RETRY_MAX_BACKOFF_SECONDS",
    "WORKER_HTTP_TIMEOUT_SECONDS",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every settings var from the env before each test."""
    for var in _ALL_VARS:
        monkeypatch.delenv(var, raising=False)


def _settings() -> WorkerSettings:
    """Build Settings without reading any on-disk ``.env`` file."""
    return WorkerSettings(_env_file=None)  # type: ignore[call-arg]


def test_defaults_load() -> None:
    s = _settings()
    assert s.log_level == "INFO"
    assert s.redis_url.startswith("redis://")
    assert s.model_service_url.startswith("http://")
    assert s.max_retries == 3
    assert s.backoff_base_seconds == 2.0
    assert s.retry_max_backoff_seconds == 60.0
    assert s.http_timeout_seconds == 10.0


def test_bare_redis_url_is_picked_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """``REDIS_URL`` (no prefix) is honoured — that's what compose sets."""
    monkeypatch.setenv("REDIS_URL", "redis://example:6379/2")
    assert _settings().redis_url == "redis://example:6379/2"


def test_prefixed_redis_url_also_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """``WORKER_REDIS_URL`` is also honoured (prefix-consistent form)."""
    monkeypatch.setenv("WORKER_REDIS_URL", "redis://prefixed:6379/3")
    assert _settings().redis_url == "redis://prefixed:6379/3"


def test_bare_database_url_is_picked_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h:5432/db")
    assert _settings().database_url == "postgresql+asyncpg://u:p@h:5432/db"


def test_max_retries_parsed_as_int(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_MAX_RETRIES", "5")
    assert _settings().max_retries == 5


def test_backoff_base_parsed_as_float(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_BACKOFF_BASE_SECONDS", "1.5")
    assert _settings().backoff_base_seconds == 1.5


def test_redact_url_strips_password() -> None:
    redacted = redact_url("postgresql://user:secret@host:5432/db")
    assert "secret" not in redacted
    assert redacted == "postgresql://user:***@host:5432/db"


def test_redact_url_passes_through_when_no_credentials() -> None:
    assert redact_url("redis://localhost:6379/0") == "redis://localhost:6379/0"
