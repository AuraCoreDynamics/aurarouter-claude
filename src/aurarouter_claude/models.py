"""Claude model catalog -- IDs, context windows, and pricing."""

from __future__ import annotations

CLAUDE_MODELS: list[dict] = [
    {
        "model_id": "claude-opus-4-20250514",
        "display_name": "Claude Opus 4",
        "context_window": 200_000,
        "cost_per_1m_input": 15.0,
        "cost_per_1m_output": 75.0,
        "capabilities": ["code", "reasoning", "chat"],
    },
    {
        "model_id": "claude-sonnet-4-20250514",
        "display_name": "Claude Sonnet 4",
        "context_window": 200_000,
        "cost_per_1m_input": 3.0,
        "cost_per_1m_output": 15.0,
        "capabilities": ["code", "reasoning", "chat"],
    },
    {
        "model_id": "claude-haiku-4-5-20251001",
        "display_name": "Claude Haiku 4.5",
        "context_window": 200_000,
        "cost_per_1m_input": 0.80,
        "cost_per_1m_output": 4.0,
        "capabilities": ["code", "chat"],
    },
]


def get_model_info(model_id: str) -> dict | None:
    """Look up a model by its ID. Returns None if not found."""
    for model in CLAUDE_MODELS:
        if model["model_id"] == model_id:
            return model
    return None


def get_default_model_id() -> str:
    """Return the default model ID (Sonnet 4)."""
    return "claude-sonnet-4-20250514"
