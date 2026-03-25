"""MCP server factory for aurarouter-claude."""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from aurarouter_claude.provider import ClaudeProvider


def create_server(api_key: str | None = None) -> FastMCP:
    """Create and return a configured FastMCP server instance.

    Parameters
    ----------
    api_key:
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
    """
    mcp = FastMCP("aurarouter-claude")
    provider = ClaudeProvider(api_key=api_key)

    @mcp.tool(name="provider.generate")
    def generate(
        prompt: str, model: str = "", json_mode: bool = False
    ) -> str:
        """Single-shot text generation via Anthropic Claude."""
        result = provider.generate(prompt, model=model, json_mode=json_mode)
        return json.dumps(result)

    @mcp.tool(name="provider.list_models")
    def list_models() -> str:
        """List available Claude models with metadata."""
        models = provider.list_models()
        return json.dumps(models)

    @mcp.tool(name="provider.generate_with_history")
    def generate_with_history(
        messages: str,
        system_prompt: str = "",
        model: str = "",
        json_mode: bool = False,
    ) -> str:
        """Multi-turn generation with message history.

        The *messages* parameter is a JSON string encoding a list of
        ``{"role": "user"|"assistant", "content": "..."}`` dicts.
        """
        parsed_messages: list[dict[str, Any]] = json.loads(messages)
        result = provider.generate_with_history(
            parsed_messages,
            system_prompt=system_prompt,
            model=model,
            json_mode=json_mode,
        )
        return json.dumps(result)

    @mcp.tool(name="provider.capabilities")
    def capabilities() -> str:
        """Advertise provider capabilities."""
        caps = [
            "provider.generate",
            "provider.list_models",
            "provider.generate_with_history",
            "provider.capabilities",
        ]
        return json.dumps(caps)

    return mcp
