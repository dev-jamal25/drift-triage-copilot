"""Tests for comms node with LLM integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent.app.core.config import AgentSettings
from agent.app.nodes.comms import comms_node
from agent.app.schemas.state import AgentState


@pytest.mark.asyncio
async def test_comms_deterministic_summary():
    """Test deterministic summary generation."""
    state: AgentState = {
        "investigation_id": "test-inv-1",
        "model_name": "test-model",
        "model_version": "v1",
        "severity": "HIGH",
        "reasoning": "Age distribution shifted significantly",
        "action_type": "retrain",
    }

    result = await comms_node(state)

    assert result["status"] == "complete"
    assert "test-model" in result["summary"]
    assert "HIGH" in result["summary"]
    assert "retrain" in result["summary"].lower()


@pytest.mark.asyncio
async def test_comms_deterministic_no_action():
    """Test deterministic summary when no action."""
    state: AgentState = {
        "investigation_id": "test-inv-2",
        "model_name": "test-model",
        "model_version": "v1",
        "severity": "LOW",
        "reasoning": "Minor drift",
        "action_type": "none",
    }

    result = await comms_node(state)

    assert result["status"] == "complete"
    assert "test-model" in result["summary"]
    assert "LOW" in result["summary"]


@pytest.mark.asyncio
async def test_comms_llm_valid_summary():
    """Test LLM generates valid summary."""
    settings = AgentSettings(use_llm=True, llm_timeout_seconds=5.0)

    state: AgentState = {
        "investigation_id": "test-inv-3",
        "_settings": settings,
        "model_name": "test-model",
        "model_version": "v1",
        "severity": "MEDIUM",
        "reasoning": "Age feature drifted",
        "action_type": "replay_test",
    }

    llm_summary = "Model 'test-model' shows moderate drift in age feature. We recommend testing the model on recent data before deploying changes. A human reviewer will approve next steps."

    mock_response = MagicMock()
    mock_response.content = llm_summary

    with patch("agent.app.nodes.comms.get_comms_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = mock_response
        mock_get_model.return_value = mock_model

        result = await comms_node(state)

    assert result["status"] == "complete"
    assert result["summary"] == llm_summary
    assert "test-model" in result["summary"]


@pytest.mark.asyncio
async def test_comms_llm_empty_response_fallback():
    """Test comms falls back on empty LLM response."""
    settings = AgentSettings(use_llm=True, llm_timeout_seconds=5.0)

    state: AgentState = {
        "investigation_id": "test-inv-4",
        "_settings": settings,
        "model_name": "test-model",
        "model_version": "v1",
        "severity": "HIGH",
        "reasoning": "Severe drift detected",
        "action_type": "retrain",
    }

    mock_response = MagicMock()
    mock_response.content = "   "  # Empty/whitespace

    with patch("agent.app.nodes.comms.get_comms_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = mock_response
        mock_get_model.return_value = mock_model

        result = await comms_node(state)

    assert result["status"] == "complete"
    # Should fall back to deterministic summary
    assert "test-model" in result["summary"]


@pytest.mark.asyncio
async def test_comms_llm_api_error_fallback():
    """Test comms falls back on API error."""
    settings = AgentSettings(use_llm=True, llm_timeout_seconds=5.0)

    state: AgentState = {
        "investigation_id": "test-inv-5",
        "_settings": settings,
        "model_name": "test-model",
        "model_version": "v1",
        "severity": "MEDIUM",
        "reasoning": "Moderate drift",
        "action_type": "replay_test",
    }

    with patch("agent.app.nodes.comms.get_comms_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.ainvoke.side_effect = Exception("API timeout")
        mock_get_model.return_value = mock_model

        result = await comms_node(state)

    assert result["status"] == "complete"
    # Should fall back to deterministic summary
    assert "test-model" in result["summary"]


@pytest.mark.asyncio
async def test_comms_llm_disabled():
    """Test comms uses deterministic when LLM disabled."""
    settings = AgentSettings(use_llm=False)

    state: AgentState = {
        "investigation_id": "test-inv-6",
        "_settings": settings,
        "model_name": "test-model",
        "model_version": "v1",
        "severity": "HIGH",
        "reasoning": "High severity drift",
        "action_type": "retrain",
    }

    with patch("agent.app.nodes.comms.get_comms_model") as mock_get_model:
        result = await comms_node(state)
        mock_get_model.assert_not_called()

    assert result["status"] == "complete"
    assert "test-model" in result["summary"]


@pytest.mark.asyncio
async def test_comms_no_settings():
    """Test comms works without settings (uses deterministic)."""
    state: AgentState = {
        "investigation_id": "test-inv-7",
        "model_name": "test-model",
        "model_version": "v1",
        "severity": "MEDIUM",
        "reasoning": "Moderate drift",
        "action_type": "replay_test",
    }

    result = await comms_node(state)

    assert result["status"] == "complete"
    assert "test-model" in result["summary"]
    assert "MEDIUM" in result["summary"]


@pytest.mark.asyncio
async def test_comms_llm_with_rollback_action():
    """Test comms with rollback action in summary."""
    settings = AgentSettings(use_llm=True, llm_timeout_seconds=5.0)

    state: AgentState = {
        "investigation_id": "test-inv-8",
        "_settings": settings,
        "model_name": "test-model",
        "model_version": "v1",
        "severity": "HIGH",
        "reasoning": "Model is unsafe",
        "action_type": "rollback",
    }

    llm_summary = "Critical drift detected. Rolling back to previous model version. This action requires human approval."

    mock_response = MagicMock()
    mock_response.content = llm_summary

    with patch("agent.app.nodes.comms.get_comms_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = mock_response
        mock_get_model.return_value = mock_model

        result = await comms_node(state)

    assert result["status"] == "complete"
    assert "Critical drift" in result["summary"]
    assert "human approval" in result["summary"].lower()


@pytest.mark.asyncio
async def test_comms_multiline_llm_summary():
    """Test comms handles multiline LLM summary."""
    settings = AgentSettings(use_llm=True, llm_timeout_seconds=5.0)

    state: AgentState = {
        "investigation_id": "test-inv-9",
        "_settings": settings,
        "model_name": "test-model",
        "model_version": "v1",
        "severity": "HIGH",
        "reasoning": "Multiple features drifted",
        "action_type": "retrain",
    }

    llm_summary = """Model 'test-model' has detected significant drift.

Key changes:
- Age distribution shifted 25% from baseline
- Job category distribution changed 18%

We recommend immediately retraining the model with recent data.
A human reviewer will validate and approve the retrain action."""

    mock_response = MagicMock()
    mock_response.content = llm_summary

    with patch("agent.app.nodes.comms.get_comms_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = mock_response
        mock_get_model.return_value = mock_model

        result = await comms_node(state)

    assert result["status"] == "complete"
    assert len(result["summary"]) > len(state["reasoning"])
    assert "Age distribution" in result["summary"]
