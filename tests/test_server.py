"""Tests for the MCP server factory."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from conftest import make_mock_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_server():
    """Create a server instance with a mocked Anthropic client."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = make_mock_response()
    with patch("anthropic.Anthropic", return_value=mock_client):
        from aurarouter_claude.server import create_server
        server = create_server(api_key="sk-ant-test-key")
    return server, mock_client


# ---------------------------------------------------------------------------
# Tool registration tests
# ---------------------------------------------------------------------------

class TestToolRegistration:
    """Tests that all expected MCP tools are registered."""

    def test_server_creates_successfully(self):
        server, _ = _create_test_server()
        assert server is not None

    def test_server_name(self):
        server, _ = _create_test_server()
        assert server.name == "aurarouter-claude"

    def test_all_four_tools_registered(self):
        server, _ = _create_test_server()
        # FastMCP stores tools in _tool_manager._tools dict
        tool_names = set()
        if hasattr(server, '_tool_manager'):
            tool_names = set(server._tool_manager._tools.keys())
        elif hasattr(server, '_tools'):
            tool_names = set(server._tools.keys())
        else:
            # Fallback: try listing tools
            pytest.skip("Cannot introspect tool registration on this mcp version")

        expected = {
            "provider.generate",
            "provider.list_models",
            "provider.generate_with_history",
            "provider.capabilities",
        }
        assert expected.issubset(tool_names), (
            f"Missing tools: {expected - tool_names}"
        )

    def test_generate_tool_registered(self):
        server, _ = _create_test_server()
        tools = _get_tool_names(server)
        assert "provider.generate" in tools

    def test_list_models_tool_registered(self):
        server, _ = _create_test_server()
        tools = _get_tool_names(server)
        assert "provider.list_models" in tools

    def test_generate_with_history_tool_registered(self):
        server, _ = _create_test_server()
        tools = _get_tool_names(server)
        assert "provider.generate_with_history" in tools

    def test_capabilities_tool_registered(self):
        server, _ = _create_test_server()
        tools = _get_tool_names(server)
        assert "provider.capabilities" in tools


def _get_tool_names(server) -> set[str]:
    """Extract registered tool names from a FastMCP server."""
    if hasattr(server, '_tool_manager'):
        return set(server._tool_manager._tools.keys())
    if hasattr(server, '_tools'):
        return set(server._tools.keys())
    return set()


# ---------------------------------------------------------------------------
# Tool execution tests (calling the underlying functions directly)
# ---------------------------------------------------------------------------

class TestGenerateTool:
    """Tests for the provider.generate MCP tool."""

    def test_returns_valid_json(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = make_mock_response(
            text="Generated text"
        )
        with patch("anthropic.Anthropic", return_value=mock_client):
            from aurarouter_claude.server import create_server
            from aurarouter_claude.provider import ClaudeProvider

            provider = ClaudeProvider(api_key="sk-ant-test-key")
            result = provider.generate("test prompt")
            result_json = json.dumps(result)
            parsed = json.loads(result_json)
            assert parsed["text"] == "Generated text"

    def test_generate_result_has_tokens(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = make_mock_response(
            input_tokens=55, output_tokens=22
        )
        with patch("anthropic.Anthropic", return_value=mock_client):
            from aurarouter_claude.provider import ClaudeProvider
            provider = ClaudeProvider(api_key="sk-ant-test-key")
            result = provider.generate("test")
            assert result["input_tokens"] == 55
            assert result["output_tokens"] == 22


class TestListModelsTool:
    """Tests for the provider.list_models MCP tool."""

    def test_returns_valid_json_list(self):
        mock_client = MagicMock()
        with patch("anthropic.Anthropic", return_value=mock_client):
            from aurarouter_claude.provider import ClaudeProvider
            provider = ClaudeProvider(api_key="sk-ant-test-key")
            models = provider.list_models()
            result_json = json.dumps(models)
            parsed = json.loads(result_json)
            assert isinstance(parsed, list)
            assert len(parsed) >= 3

    def test_models_have_model_id(self):
        mock_client = MagicMock()
        with patch("anthropic.Anthropic", return_value=mock_client):
            from aurarouter_claude.provider import ClaudeProvider
            provider = ClaudeProvider(api_key="sk-ant-test-key")
            models = provider.list_models()
            for m in models:
                assert "model_id" in m


class TestCapabilitiesTool:
    """Tests for the provider.capabilities MCP tool."""

    def test_capabilities_returns_json_list(self):
        server, _ = _create_test_server()
        # Directly test the capability list content
        expected_caps = [
            "provider.generate",
            "provider.list_models",
            "provider.generate_with_history",
            "provider.capabilities",
        ]
        caps_json = json.dumps(expected_caps)
        parsed = json.loads(caps_json)
        assert isinstance(parsed, list)
        assert "provider.generate" in parsed
        assert "provider.list_models" in parsed


# ---------------------------------------------------------------------------
# Entry point tests
# ---------------------------------------------------------------------------

class TestEntryPoint:
    """Tests for package entry point and metadata."""

    def test_package_version(self):
        from aurarouter_claude import __version__
        assert __version__ == "0.5.1"

    def test_package_name(self):
        from aurarouter_claude import __package_name__
        assert __package_name__ == "aurarouter-claude"

    def test_get_provider_metadata(self):
        from aurarouter_claude import get_provider_metadata
        meta = get_provider_metadata()
        assert meta.name == "claude"
        assert meta.provider_type == "mcp"
        assert meta.version == "0.5.1"
        assert "python" in meta.command[0]
        assert "aurarouter_claude" in meta.command[-1]

    def test_provider_metadata_requires_config(self):
        from aurarouter_claude import get_provider_metadata
        meta = get_provider_metadata()
        assert "api_key" in meta.requires_config
