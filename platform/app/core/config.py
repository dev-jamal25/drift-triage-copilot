"""Platform service settings.

All runtime config flows through this single ``Settings`` class. Per CLAUDE.md
hard rules, ``os.getenv`` may not be called outside this module.

Env vars are picked up with the ``PLATFORM_`` prefix (e.g.
``PLATFORM_MLFLOW_TRACKING_URI``). A local ``.env`` file is also honoured.

``database_url`` is the only field that breaks the prefix convention: it
accepts the project-wide ``DATABASE_URL`` (used in docker-compose) as well
as ``PLATFORM_DATABASE_URL``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the platform model service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PLATFORM_",
        case_sensitive=False,
        extra="ignore",
        # ``model_*`` field names collide with Pydantic's reserved namespace;
        # disable the protection so we can keep familiar names.
        protected_namespaces=(),
    )

    mlflow_tracking_uri: str = "file:./mlruns"
    model_name: str = "bank-marketing-classifier"
    model_alias: str = "staging"
    load_model_on_startup: bool = True
    log_level: str = "INFO"

    # Project-wide convention: services share one ``DATABASE_URL``. We accept
    # ``PLATFORM_DATABASE_URL`` too for symmetry with the rest of this Settings
    # class; whichever is set wins (``DATABASE_URL`` is checked first).
    database_url: str = Field(
        default="postgresql+asyncpg://drift_user:change_me_locally@localhost:5432/drift_triage",
        validation_alias=AliasChoices("DATABASE_URL", "PLATFORM_DATABASE_URL"),
    )
