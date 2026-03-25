"""Shared fixtures for aurarouter-claude tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock response objects matching the Anthropic SDK structure
# ---------------------------------------------------------------------------

@dataclass
class MockTextBlock:
    """Mimics anthropic.types.TextBlock."""
    text: str
    type: str = "text"


@dataclass
class MockUsage:
    """Mimics anthropic.types.Usage."""
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class MockMessage:
    """Mimics anthropic.types.Message."""
    id: str = "msg_test123"
    type: str = "message"
    role: str = "assistant"
    content: list[MockTextBlock] = field(default_factory=lambda: [MockTextBlock(text="Hello, world!")])
    model: str = "claude-sonnet-4-20250514"
    usage: MockUsage = field(default_factory=MockUsage)
    stop_reason: str = "end_turn"


def make_mock_response(
    text: str = "Hello, world!",
    input_tokens: int = 100,
    output_tokens: int = 50,
    model: str = "claude-sonnet-4-20250514",
) -> MockMessage:
    """Create a mock Anthropic message response."""
    return MockMessage(
        content=[MockTextBlock(text=text)],
        model=model,
        usage=MockUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


@pytest.fixture
def mock_anthropic_client():
    """Fixture providing a mocked anthropic.Anthropic client.

    Usage::

        def test_something(mock_anthropic_client):
            client, mock_create = mock_anthropic_client
            mock_create.return_value = make_mock_response(text="test output")
            # ... use provider ...
    """
    mock_client = MagicMock()
    mock_create = mock_client.messages.create
    mock_create.return_value = make_mock_response()

    with patch("anthropic.Anthropic", return_value=mock_client):
        yield mock_client, mock_create


@pytest.fixture
def provider(mock_anthropic_client):
    """Fixture providing a ClaudeProvider with a mocked Anthropic client."""
    from aurarouter_claude.provider import ClaudeProvider

    _client, _mock_create = mock_anthropic_client
    p = ClaudeProvider(api_key="sk-ant-test-key")
    return p


@pytest.fixture
def provider_and_mock(mock_anthropic_client):
    """Fixture providing both a ClaudeProvider and the mock create callable."""
    from aurarouter_claude.provider import ClaudeProvider

    client, mock_create = mock_anthropic_client
    p = ClaudeProvider(api_key="sk-ant-test-key")
    return p, mock_create
