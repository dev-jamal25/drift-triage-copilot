"""LLM client factories for Anthropic Claude models."""

from __future__ import annotations

from agent.app.core.config import AgentSettings
from langchain_anthropic import ChatAnthropic


def get_triage_model(settings: AgentSettings) -> ChatAnthropic:
    """Create a ChatAnthropic client configured for triage classification.

    Args:
        settings: Agent settings with API key and model name.

    Returns:
        ChatAnthropic instance with triage model (Haiku).
    """
    return ChatAnthropic(
        api_key=settings.anthropic_api_key.get_secret_value(),
        model_name=settings.llm_triage_model,
        temperature=0.0,
        timeout=settings.llm_timeout_seconds,
    )


def get_action_model(settings: AgentSettings) -> ChatAnthropic:
    """Create a ChatAnthropic client configured for action selection.

    Args:
        settings: Agent settings with API key and model name.

    Returns:
        ChatAnthropic instance with action model (Sonnet).
    """
    return ChatAnthropic(
        api_key=settings.anthropic_api_key.get_secret_value(),
        model_name=settings.llm_action_model,
        temperature=0.0,
        timeout=settings.llm_timeout_seconds,
    )


def get_comms_model(settings: AgentSettings) -> ChatAnthropic:
    """Create a ChatAnthropic client configured for summary generation.

    Args:
        settings: Agent settings with API key and model name.

    Returns:
        ChatAnthropic instance with comms model (Haiku).
    """
    return ChatAnthropic(
        api_key=settings.anthropic_api_key.get_secret_value(),
        model_name=settings.llm_comms_model,
        temperature=0.0,
        timeout=settings.llm_timeout_seconds,
    )
