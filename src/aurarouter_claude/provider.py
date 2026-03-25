"""ClaudeProvider -- wraps the Anthropic Python SDK for generation."""

from __future__ import annotations

import os
from typing import Any

import anthropic

from aurarouter_claude.models import CLAUDE_MODELS, get_default_model_id, get_model_info


class ClaudeProvider:
    """Thin wrapper around the Anthropic messages API.

    Parameters
    ----------
    api_key:
        Anthropic API key.  If *None*, the SDK resolves from the
        ``ANTHROPIC_API_KEY`` environment variable automatically.
    default_model:
        Model ID to use when the caller does not specify one.
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "claude-sonnet-4-20250514",
    ) -> None:
        # Validate that an API key is available (either passed or in env)
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Anthropic API key provided. Set the ANTHROPIC_API_KEY "
                "environment variable or pass api_key= to ClaudeProvider."
            )
        self._client = anthropic.Anthropic(api_key=resolved_key)
        self._default_model = default_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        model: str = "",
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """Single-shot text generation.

        Returns
        -------
        dict with keys: text, input_tokens, output_tokens, model_id
        """
        model_id = model or self._default_model
        system_parts: list[str] = []

        if json_mode:
            system_parts.append("Respond in valid JSON.")

        system_prompt = "\n".join(system_parts) if system_parts else ""

        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)

        text = self._extract_text(response)
        return {
            "text": text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "model_id": model_id,
        }

    def generate_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        model: str = "",
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """Multi-turn generation using the native messages API.

        Parameters
        ----------
        messages:
            List of ``{"role": "user"|"assistant", "content": str}`` dicts.
        system_prompt:
            Optional system instruction.
        model:
            Model ID override.
        json_mode:
            If True, appends JSON instruction to the system prompt.

        Returns
        -------
        dict with keys: text, input_tokens, output_tokens, model_id
        """
        model_id = model or self._default_model

        # Build system prompt
        system_parts: list[str] = []
        if system_prompt:
            system_parts.append(system_prompt)
        if json_mode:
            system_parts.append("Respond in valid JSON.")
        combined_system = "\n".join(system_parts) if system_parts else ""

        # Sanitize messages — Anthropic expects role to be "user" or "assistant"
        sanitized: list[dict[str, str]] = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                # Fold system messages into the system prompt
                if combined_system:
                    combined_system += "\n" + msg.get("content", "")
                else:
                    combined_system = msg.get("content", "")
                continue
            sanitized.append({
                "role": role,
                "content": msg.get("content", ""),
            })

        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": 4096,
            "messages": sanitized,
        }
        if combined_system:
            kwargs["system"] = combined_system

        response = self._client.messages.create(**kwargs)

        text = self._extract_text(response)
        return {
            "text": text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "model_id": model_id,
        }

    def list_models(self) -> list[dict]:
        """Return available Claude models with metadata."""
        return [dict(m) for m in CLAUDE_MODELS]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract text from an Anthropic message response."""
        parts: list[str] = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)
