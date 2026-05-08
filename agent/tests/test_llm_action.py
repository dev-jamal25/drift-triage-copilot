"""Tests for action node with LLM integration and safety constraints."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent.app.core.config import AgentSettings
from agent.app.nodes.action import action_node
from agent.app.schemas.state import AgentState


@pytest.mark.asyncio
async def test_action_low_severity_no_action():
    """Test action skips LOW severity drift."""
    state: AgentState = {
        "investigation_id": "test-inv-1",
        "severity": "LOW",
        "model_version": "v1",
    }

    result = await action_node(state)

    assert result["action_queued"] is False
    assert result["status"] == "skipped"
    assert result.get("action_type") is None


@pytest.mark.asyncio
async def test_action_deterministic_medium():
    """Test deterministic action for MEDIUM severity."""
    state: AgentState = {
        "investigation_id": "test-inv-2",
        "severity": "MEDIUM",
        "model_version": "v1",
    }

    result = await action_node(state)

    assert result["action_queued"] is True
    assert result["status"] == "paused"
    assert result["action_type"] == "replay_test"
    assert result["idempotency_key"]


@pytest.mark.asyncio
async def test_action_deterministic_high():
    """Test deterministic action for HIGH severity."""
    state: AgentState = {
        "investigation_id": "test-inv-3",
        "severity": "HIGH",
        "model_version": "v1",
    }

    result = await action_node(state)

    assert result["action_queued"] is True
    assert result["status"] == "paused"
    assert result["action_type"] == "retrain"
    assert result["idempotency_key"]


@pytest.mark.asyncio
async def test_action_llm_high_retrain():
    """Test LLM recommends retrain for HIGH severity."""
    settings = AgentSettings(use_llm=True, llm_timeout_seconds=5.0)

    state: AgentState = {
        "investigation_id": "test-inv-4",
        "_settings": settings,
        "severity": "HIGH",
        "model_version": "v1",
        "drift_event": {
            "drift_report": {"overall_severity": "red"},
        },
    }

    mock_response = MagicMock()
    mock_response.content = json.dumps(
        {"action_type": "retrain", "reasoning": "Model needs retraining"}
    )

    with patch("agent.app.nodes.action.get_action_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = mock_response
        mock_get_model.return_value = mock_model

        result = await action_node(state)

    assert result["action_queued"] is True
    assert result["action_type"] == "retrain"


@pytest.mark.asyncio
async def test_action_llm_high_rollback():
    """Test LLM can recommend rollback for HIGH severity (with gating note)."""
    settings = AgentSettings(use_llm=True, llm_timeout_seconds=5.0)

    state: AgentState = {
        "investigation_id": "test-inv-5",
        "_settings": settings,
        "severity": "HIGH",
        "model_version": "v1",
        "drift_event": {
            "drift_report": {"overall_severity": "red"},
        },
    }

    mock_response = MagicMock()
    mock_response.content = json.dumps({"action_type": "rollback", "reasoning": "Model is unsafe"})

    with patch("agent.app.nodes.action.get_action_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = mock_response
        mock_get_model.return_value = mock_model

        result = await action_node(state)

    assert result["action_queued"] is True
    assert result["action_type"] == "rollback"


@pytest.mark.asyncio
async def test_action_llm_medium_only_replay_test():
    """Test LLM must use replay_test for MEDIUM severity (blocks retrain/rollback)."""
    settings = AgentSettings(use_llm=True, llm_timeout_seconds=5.0)

    state: AgentState = {
        "investigation_id": "test-inv-6",
        "_settings": settings,
        "severity": "MEDIUM",
        "model_version": "v1",
        "drift_event": {
            "drift_report": {"overall_severity": "yellow"},
        },
    }

    # LLM recommends retrain (invalid for MEDIUM)
    mock_response = MagicMock()
    mock_response.content = json.dumps({"action_type": "retrain", "reasoning": "Should retrain"})

    with patch("agent.app.nodes.action.get_action_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = mock_response
        mock_get_model.return_value = mock_model

        result = await action_node(state)

    # Should be overridden to replay_test
    assert result["action_queued"] is True
    assert result["action_type"] == "replay_test"


@pytest.mark.asyncio
async def test_action_llm_medium_blocks_rollback():
    """Test LLM cannot recommend rollback for MEDIUM severity."""
    settings = AgentSettings(use_llm=True, llm_timeout_seconds=5.0)

    state: AgentState = {
        "investigation_id": "test-inv-7",
        "_settings": settings,
        "severity": "MEDIUM",
        "model_version": "v1",
        "drift_event": {
            "drift_report": {"overall_severity": "yellow"},
        },
    }

    # LLM recommends rollback (invalid for MEDIUM)
    mock_response = MagicMock()
    mock_response.content = json.dumps({"action_type": "rollback", "reasoning": "Should rollback"})

    with patch("agent.app.nodes.action.get_action_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = mock_response
        mock_get_model.return_value = mock_model

        result = await action_node(state)

    # Should be overridden to replay_test
    assert result["action_queued"] is True
    assert result["action_type"] == "replay_test"


@pytest.mark.asyncio
async def test_action_llm_invalid_action_fallback():
    """Test action falls back on invalid action type."""
    settings = AgentSettings(use_llm=True, llm_timeout_seconds=5.0)

    state: AgentState = {
        "investigation_id": "test-inv-8",
        "_settings": settings,
        "severity": "HIGH",
        "model_version": "v1",
        "drift_event": {
            "drift_report": {"overall_severity": "red"},
        },
    }

    mock_response = MagicMock()
    mock_response.content = json.dumps({"action_type": "invalid_action", "reasoning": "Invalid"})

    with patch("agent.app.nodes.action.get_action_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = mock_response
        mock_get_model.return_value = mock_model

        result = await action_node(state)

    # Should fall back to deterministic (retrain for HIGH)
    assert result["action_queued"] is True
    assert result["action_type"] == "retrain"


@pytest.mark.asyncio
async def test_action_llm_invalid_json_fallback():
    """Test action falls back on invalid JSON."""
    settings = AgentSettings(use_llm=True, llm_timeout_seconds=5.0)

    state: AgentState = {
        "investigation_id": "test-inv-9",
        "_settings": settings,
        "severity": "MEDIUM",
        "model_version": "v1",
        "drift_event": {
            "drift_report": {"overall_severity": "yellow"},
        },
    }

    mock_response = MagicMock()
    mock_response.content = "Invalid JSON"

    with patch("agent.app.nodes.action.get_action_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = mock_response
        mock_get_model.return_value = mock_model

        result = await action_node(state)

    # Should fall back to deterministic (replay_test for MEDIUM)
    assert result["action_queued"] is True
    assert result["action_type"] == "replay_test"


@pytest.mark.asyncio
async def test_action_llm_api_error_fallback():
    """Test action falls back on API error."""
    settings = AgentSettings(use_llm=True, llm_timeout_seconds=5.0)

    state: AgentState = {
        "investigation_id": "test-inv-10",
        "_settings": settings,
        "severity": "HIGH",
        "model_version": "v1",
        "drift_event": {
            "drift_report": {"overall_severity": "red"},
        },
    }

    with patch("agent.app.nodes.action.get_action_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.ainvoke.side_effect = Exception("API timeout")
        mock_get_model.return_value = mock_model

        result = await action_node(state)

    # Should fall back to deterministic (retrain for HIGH)
    assert result["action_queued"] is True
    assert result["action_type"] == "retrain"


@pytest.mark.asyncio
async def test_action_llm_disabled():
    """Test action uses deterministic when LLM disabled."""
    settings = AgentSettings(use_llm=False)

    state: AgentState = {
        "investigation_id": "test-inv-11",
        "_settings": settings,
        "severity": "HIGH",
        "model_version": "v1",
    }

    with patch("agent.app.nodes.action.get_action_model") as mock_get_model:
        result = await action_node(state)
        mock_get_model.assert_not_called()

    assert result["action_queued"] is True
    assert result["action_type"] == "retrain"


@pytest.mark.asyncio
async def test_action_idempotency_key_deterministic():
    """Test idempotency key is deterministic."""
    state1: AgentState = {
        "investigation_id": "test-inv-12",
        "severity": "HIGH",
        "model_version": "v1",
    }

    state2: AgentState = {
        "investigation_id": "test-inv-12",
        "severity": "HIGH",
        "model_version": "v1",
    }

    result1 = await action_node(state1)
    result2 = await action_node(state2)

    # Same investigation, same action, same version -> same idempotency key
    assert result1["idempotency_key"] == result2["idempotency_key"]


@pytest.mark.asyncio
async def test_action_high_invalid_action_defaults_to_retrain():
    """Test HIGH severity with invalid LLM action defaults to retrain."""
    settings = AgentSettings(use_llm=True, llm_timeout_seconds=5.0)

    state: AgentState = {
        "investigation_id": "test-inv-13",
        "_settings": settings,
        "severity": "HIGH",
        "model_version": "v1",
        "drift_event": {
            "drift_report": {"overall_severity": "red"},
        },
    }

    mock_response = MagicMock()
    # LLM returns valid action_type but not in HIGH allowed set
    mock_response.content = json.dumps(
        {"action_type": "replay_test", "reasoning": "Should only replay"}
    )

    with patch("agent.app.nodes.action.get_action_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = mock_response
        mock_get_model.return_value = mock_model

        result = await action_node(state)

    # Should default to retrain for HIGH
    assert result["action_type"] == "retrain"
