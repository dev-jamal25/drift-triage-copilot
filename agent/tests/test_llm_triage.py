"""Tests for triage node with LLM integration."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent.app.core.config import AgentSettings
from agent.app.nodes.triage import triage_node
from agent.app.schemas.state import AgentState


@pytest.mark.asyncio
async def test_triage_deterministic_low_severity():
    """Test deterministic triage with LOW severity."""
    state: AgentState = {
        "investigation_id": "test-inv-1",
        "drift_event": {
            "model_name": "test-model",
            "model_version": "v1",
            "drift_report": {"overall_severity": "green"},
        },
    }

    result = await triage_node(state)

    assert result["severity"] == "LOW"
    assert "green" in result["reasoning"]


@pytest.mark.asyncio
async def test_triage_deterministic_medium_severity():
    """Test deterministic triage with MEDIUM severity."""
    state: AgentState = {
        "investigation_id": "test-inv-2",
        "drift_event": {
            "model_name": "test-model",
            "model_version": "v1",
            "drift_report": {"overall_severity": "yellow"},
        },
    }

    result = await triage_node(state)

    assert result["severity"] == "MEDIUM"
    assert "yellow" in result["reasoning"]


@pytest.mark.asyncio
async def test_triage_deterministic_high_severity():
    """Test deterministic triage with HIGH severity."""
    state: AgentState = {
        "investigation_id": "test-inv-3",
        "drift_event": {
            "model_name": "test-model",
            "model_version": "v1",
            "drift_report": {"overall_severity": "red"},
        },
    }

    result = await triage_node(state)

    assert result["severity"] == "HIGH"
    assert "red" in result["reasoning"]


@pytest.mark.asyncio
async def test_triage_no_drift_event():
    """Test triage with missing drift event."""
    state: AgentState = {"investigation_id": "test-inv-4"}

    result = await triage_node(state)

    assert result["severity"] == "LOW"
    assert "No drift event" in result["reasoning"]


@pytest.mark.asyncio
async def test_triage_llm_valid_response():
    """Test triage with valid LLM response."""
    settings = AgentSettings(use_llm=True, llm_timeout_seconds=5.0)

    state: AgentState = {
        "investigation_id": "test-inv-5",
        "_settings": settings,
        "drift_event": {
            "model_name": "test-model",
            "model_version": "v1",
            "drift_report": {
                "overall_severity": "yellow",
                "numeric_drift": {"age": 0.15},
                "categorical_drift": {"job": 0.08},
            },
        },
    }

    mock_response = MagicMock()
    mock_response.content = json.dumps(
        {"severity": "HIGH", "reasoning": "Age distribution shifted significantly"}
    )

    with patch("agent.app.nodes.triage.get_triage_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = mock_response
        mock_get_model.return_value = mock_model

        result = await triage_node(state)

    assert result["severity"] == "HIGH"
    assert "Age distribution" in result["reasoning"]


@pytest.mark.asyncio
async def test_triage_llm_invalid_json_fallback():
    """Test triage falls back to deterministic on invalid JSON."""
    settings = AgentSettings(use_llm=True, llm_timeout_seconds=5.0)

    state: AgentState = {
        "investigation_id": "test-inv-6",
        "_settings": settings,
        "drift_event": {
            "model_name": "test-model",
            "model_version": "v1",
            "drift_report": {"overall_severity": "yellow"},
        },
    }

    mock_response = MagicMock()
    mock_response.content = "Invalid JSON response"

    with patch("agent.app.nodes.triage.get_triage_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = mock_response
        mock_get_model.return_value = mock_model

        result = await triage_node(state)

    # Should fall back to deterministic
    assert result["severity"] == "MEDIUM"
    assert "fell back" in result["reasoning"]


@pytest.mark.asyncio
async def test_triage_llm_invalid_severity_fallback():
    """Test triage falls back on invalid severity value."""
    settings = AgentSettings(use_llm=True, llm_timeout_seconds=5.0)

    state: AgentState = {
        "investigation_id": "test-inv-7",
        "_settings": settings,
        "drift_event": {
            "model_name": "test-model",
            "model_version": "v1",
            "drift_report": {"overall_severity": "yellow"},
        },
    }

    mock_response = MagicMock()
    mock_response.content = json.dumps({"severity": "INVALID", "reasoning": "Invalid severity"})

    with patch("agent.app.nodes.triage.get_triage_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = mock_response
        mock_get_model.return_value = mock_model

        result = await triage_node(state)

    # Should fall back to deterministic
    assert result["severity"] == "MEDIUM"
    assert "fell back" in result["reasoning"]


@pytest.mark.asyncio
async def test_triage_llm_api_error_fallback():
    """Test triage falls back on API error."""
    settings = AgentSettings(use_llm=True, llm_timeout_seconds=5.0)

    state: AgentState = {
        "investigation_id": "test-inv-8",
        "_settings": settings,
        "drift_event": {
            "model_name": "test-model",
            "model_version": "v1",
            "drift_report": {"overall_severity": "red"},
        },
    }

    with patch("agent.app.nodes.triage.get_triage_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.ainvoke.side_effect = Exception("API timeout")
        mock_get_model.return_value = mock_model

        result = await triage_node(state)

    # Should fall back to deterministic
    assert result["severity"] == "HIGH"
    assert "fell back" in result["reasoning"]
    assert "error" in result["reasoning"].lower()


@pytest.mark.asyncio
async def test_triage_llm_disabled():
    """Test triage uses deterministic when LLM disabled."""
    settings = AgentSettings(use_llm=False)

    state: AgentState = {
        "investigation_id": "test-inv-9",
        "_settings": settings,
        "drift_event": {
            "model_name": "test-model",
            "model_version": "v1",
            "drift_report": {"overall_severity": "red"},
        },
    }

    with patch("agent.app.nodes.triage.get_triage_model") as mock_get_model:
        # Should never call get_triage_model when use_llm=False
        result = await triage_node(state)
        mock_get_model.assert_not_called()

    assert result["severity"] == "HIGH"
    assert "deterministic" in result["reasoning"]
