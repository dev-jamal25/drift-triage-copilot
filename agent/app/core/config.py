"""Agent service settings.

All runtime config flows through this single ``AgentSettings`` class. Per
CLAUDE.md hard rules, ``os.getenv`` may not be called outside this module.

Env vars are picked up with the ``AGENT_`` prefix (e.g.
``AGENT_DATABASE_URL``). A local ``.env`` file is also honoured.

Project-wide vars (``DATABASE_URL``, ``REDIS_URL``, ``MODEL_SERVICE_URL``,
``LOG_LEVEL``) accept both the bare name (used in docker-compose.yml) and
the ``AGENT_``-prefixed form, via ``AliasChoices``. The bare name is
checked first so the compose-file value wins when both are set.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Runtime configuration for the drift triage agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AGENT_",
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
        validation_alias=AliasChoices("LOG_LEVEL", "AGENT_LOG_LEVEL"),
    )

    database_url: str = Field(
        default="postgresql+asyncpg://drift_user:change_me_locally@localhost:5432/drift_triage",
        validation_alias=AliasChoices("DATABASE_URL", "AGENT_DATABASE_URL"),
    )

    redis_url: str = Field(
        default="redis://localhost:6379/0",
        validation_alias=AliasChoices("REDIS_URL", "AGENT_REDIS_URL"),
    )

    model_service_url: str = Field(
        default="http://localhost:8000",
        validation_alias=AliasChoices("MODEL_SERVICE_URL", "AGENT_MODEL_SERVICE_URL"),
    )

    anthropic_api_key: SecretStr = Field(
        default="sk-ant-PLACEHOLDER",
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "AGENT_ANTHROPIC_API_KEY"),
    )

    llm_triage_model: str = Field(
        default="claude-haiku-4-5-20251001",
        validation_alias=AliasChoices("LLM_TRIAGE_MODEL", "AGENT_LLM_TRIAGE_MODEL"),
    )

    llm_action_model: str = Field(
        default="claude-sonnet-4-6",
        validation_alias=AliasChoices("LLM_ACTION_MODEL", "AGENT_LLM_ACTION_MODEL"),
    )

    llm_comms_model: str = Field(
        default="claude-haiku-4-5-20251001",
        validation_alias=AliasChoices("LLM_COMMS_MODEL", "AGENT_LLM_COMMS_MODEL"),
    )

    use_llm: bool = Field(
        default=False,
        validation_alias=AliasChoices("AGENT_USE_LLM"),
        description="Enable LLM inference; defaults to deterministic fallback",
    )

    llm_timeout_seconds: float = Field(
        default=20.0,
        validation_alias=AliasChoices("AGENT_LLM_TIMEOUT_SECONDS"),
        gt=0,
        description="Timeout for LLM API calls in seconds",
    )


def redact_url(url: str) -> str:
    """Drop the password from a connection URL before logging.

    Used by ``database.py`` and ``queue/client.py`` to redact credentials.
    """
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host = rest.split("@", 1)
    if ":" in creds:
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"
    return url
