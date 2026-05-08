"""Worker service settings.

All runtime config flows through this single ``WorkerSettings`` class. Per
CLAUDE.md hard rules, ``os.getenv`` may not be called outside this module.

Env vars are picked up with the ``WORKER_`` prefix (e.g.
``WORKER_HTTP_TIMEOUT_SECONDS``). A local ``.env`` file is also honoured.

Project-wide vars (``DATABASE_URL``, ``REDIS_URL``, ``MODEL_SERVICE_URL``,
``LOG_LEVEL``) accept both the bare name (used in docker-compose.yml) and
the ``WORKER_``-prefixed form, via ``AliasChoices``. The bare name is
checked first so the compose-file value wins when both are set.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    """Runtime configuration for the action worker."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="WORKER_",
        case_sensitive=False,
        extra="ignore",
        # ``model_*`` would collide with Pydantic's reserved namespace; disable
        # so we can keep familiar field names if added later.
        protected_namespaces=(),
        # Required so fields with ``validation_alias=AliasChoices(...)`` also
        # accept the field name as a kwarg — tests rely on the kwarg form.
        populate_by_name=True,
    )

    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("LOG_LEVEL", "WORKER_LOG_LEVEL"),
    )

    database_url: str = Field(
        default="postgresql+asyncpg://drift_user:change_me_locally@localhost:5432/drift_triage",
        validation_alias=AliasChoices("DATABASE_URL", "WORKER_DATABASE_URL"),
    )

    redis_url: str = Field(
        default="redis://localhost:6379/0",
        validation_alias=AliasChoices("REDIS_URL", "WORKER_REDIS_URL"),
    )

    model_service_url: str = Field(
        default="http://localhost:8000",
        validation_alias=AliasChoices("MODEL_SERVICE_URL", "WORKER_MODEL_SERVICE_URL"),
    )

    # Retry policy — see DECISIONS.md "Idempotency strategy in the worker".
    max_retries: int = Field(default=3, ge=1)
    backoff_base_seconds: float = Field(default=2.0, gt=0)
    retry_max_backoff_seconds: float = Field(default=60.0, gt=0)

    # HTTP client default for outgoing calls (e.g. to model-service).
    http_timeout_seconds: float = Field(default=10.0, gt=0)


def redact_url(url: str) -> str:
    """Drop the password from a connection URL before logging.

    Used by ``main`` (Redis URL) and (after Step 6) ``db.engine`` (DB URL).
    Mirrors ``platform/app/db/engine.py:_redact``.
    """
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host = rest.split("@", 1)
    if ":" in creds:
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"
    return url
