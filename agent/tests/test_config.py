"""Tests for AgentSettings configuration."""

from agent.app.core.config import AgentSettings, redact_url


class TestAgentSettings:
    """Test AgentSettings loading and validation."""

    def test_load_defaults(self) -> None:
        """Test that AgentSettings loads with defaults."""
        settings = AgentSettings()
        assert settings.log_level == "INFO"
        assert "drift_user" in settings.database_url
        assert settings.redis_url.startswith("redis://")

    def test_database_url_from_env(self, monkeypatch) -> None:
        """Test that DATABASE_URL env var is picked up."""
        test_url = "postgresql+asyncpg://testuser:testpass@testhost:5432/testdb"
        monkeypatch.setenv("DATABASE_URL", test_url)
        settings = AgentSettings()
        assert settings.database_url == test_url

    def test_database_url_with_agent_prefix(self, monkeypatch) -> None:
        """Test that AGENT_DATABASE_URL takes precedence over DATABASE_URL."""
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user1@host1/db1")
        monkeypatch.setenv("AGENT_DATABASE_URL", "postgresql+asyncpg://user2@host2/db2")
        settings = AgentSettings()
        # DATABASE_URL should be checked first per AliasChoices order
        assert settings.database_url == "postgresql+asyncpg://user1@host1/db1"

    def test_redis_url_from_env(self, monkeypatch) -> None:
        """Test that REDIS_URL env var is picked up."""
        test_url = "redis://testhost:6379/0"
        monkeypatch.setenv("REDIS_URL", test_url)
        settings = AgentSettings()
        assert settings.redis_url == test_url

    def test_log_level_from_env(self, monkeypatch) -> None:
        """Test that LOG_LEVEL env var is picked up."""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        settings = AgentSettings()
        assert settings.log_level == "DEBUG"

    def test_llm_models_from_env(self, monkeypatch) -> None:
        """Test that LLM model env vars are picked up."""
        monkeypatch.setenv("LLM_TRIAGE_MODEL", "claude-3-sonnet")
        monkeypatch.setenv("LLM_ACTION_MODEL", "claude-3-opus")
        settings = AgentSettings()
        assert settings.llm_triage_model == "claude-3-sonnet"
        assert settings.llm_action_model == "claude-3-opus"

    def test_anthropic_api_key(self, monkeypatch) -> None:
        """Test that ANTHROPIC_API_KEY is loaded."""
        test_key = "sk-ant-test123"
        monkeypatch.setenv("ANTHROPIC_API_KEY", test_key)
        settings = AgentSettings()
        # SecretStr.get_secret_value() returns the actual value
        assert settings.anthropic_api_key.get_secret_value() == test_key

    def test_model_service_url_from_env(self, monkeypatch) -> None:
        """Test that MODEL_SERVICE_URL env var is picked up."""
        test_url = "http://test-model:9000"
        monkeypatch.setenv("MODEL_SERVICE_URL", test_url)
        settings = AgentSettings()
        assert settings.model_service_url == test_url


class TestRedactUrl:
    """Test the redact_url utility function."""

    def test_redact_postgres_url(self) -> None:
        """Test redacting a PostgreSQL connection URL."""
        url = "postgresql+asyncpg://drift_user:secretpassword@postgres:5432/drift_triage"
        redacted = redact_url(url)
        assert "drift_user:***@postgres" in redacted
        assert "secretpassword" not in redacted

    def test_redact_redis_url(self) -> None:
        """Test redacting a Redis URL."""
        url = "redis://localhost:6379/0"
        redacted = redact_url(url)
        assert redacted == url  # No credentials, should be unchanged

    def test_redact_url_no_scheme(self) -> None:
        """Test redacting a URL without a scheme."""
        url = "localhost:5432"
        redacted = redact_url(url)
        assert redacted == url

    def test_redact_url_no_at_sign(self) -> None:
        """Test redacting a URL without @ sign."""
        url = "postgresql://localhost:5432/drift_triage"
        redacted = redact_url(url)
        assert redacted == url
