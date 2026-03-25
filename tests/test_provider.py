"""Tests for ClaudeProvider."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from conftest import MockMessage, MockTextBlock, MockUsage, make_mock_response


class TestClaudeProviderInit:
    """Tests for ClaudeProvider initialization."""

    def test_init_with_api_key(self, mock_anthropic_client):
        from aurarouter_claude.provider import ClaudeProvider
        provider = ClaudeProvider(api_key="sk-ant-test-key")
        assert provider._default_model == "claude-sonnet-4-20250514"

    def test_init_with_env_key(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-env-key"}):
            with patch("anthropic.Anthropic") as mock_cls:
                from aurarouter_claude.provider import ClaudeProvider
                provider = ClaudeProvider()
                mock_cls.assert_called_once_with(api_key="sk-ant-env-key")

    def test_init_missing_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("ANTHROPIC_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                from aurarouter_claude.provider import ClaudeProvider
                with pytest.raises(ValueError, match="No Anthropic API key"):
                    ClaudeProvider()

    def test_init_custom_default_model(self, mock_anthropic_client):
        from aurarouter_claude.provider import ClaudeProvider
        provider = ClaudeProvider(
            api_key="sk-ant-test-key",
            default_model="claude-opus-4-20250514",
        )
        assert provider._default_model == "claude-opus-4-20250514"


class TestGenerate:
    """Tests for ClaudeProvider.generate()."""

    def test_generate_returns_text(self, provider_and_mock):
        provider, mock_create = provider_and_mock
        mock_create.return_value = make_mock_response(text="Test response")
        result = provider.generate("Hello")
        assert result["text"] == "Test response"

    def test_generate_returns_token_usage(self, provider_and_mock):
        provider, mock_create = provider_and_mock
        mock_create.return_value = make_mock_response(
            input_tokens=42, output_tokens=17
        )
        result = provider.generate("Hello")
        assert result["input_tokens"] == 42
        assert result["output_tokens"] == 17

    def test_generate_returns_model_id(self, provider_and_mock):
        provider, mock_create = provider_and_mock
        result = provider.generate("Hello")
        assert result["model_id"] == "claude-sonnet-4-20250514"

    def test_generate_with_explicit_model(self, provider_and_mock):
        provider, mock_create = provider_and_mock
        result = provider.generate("Hello", model="claude-opus-4-20250514")
        assert result["model_id"] == "claude-opus-4-20250514"
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-20250514"

    def test_generate_passes_prompt_as_user_message(self, provider_and_mock):
        provider, mock_create = provider_and_mock
        provider.generate("What is 2+2?")
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["messages"] == [
            {"role": "user", "content": "What is 2+2?"}
        ]

    def test_generate_json_mode_adds_system_prompt(self, provider_and_mock):
        provider, mock_create = provider_and_mock
        provider.generate("List items", json_mode=True)
        call_kwargs = mock_create.call_args[1]
        assert "system" in call_kwargs
        assert "Respond in valid JSON" in call_kwargs["system"]

    def test_generate_no_json_mode_no_system(self, provider_and_mock):
        provider, mock_create = provider_and_mock
        provider.generate("Hello")
        call_kwargs = mock_create.call_args[1]
        assert "system" not in call_kwargs or call_kwargs.get("system") == ""

    def test_generate_sets_max_tokens(self, provider_and_mock):
        provider, mock_create = provider_and_mock
        provider.generate("Hello")
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["max_tokens"] == 4096


class TestGenerateWithHistory:
    """Tests for ClaudeProvider.generate_with_history()."""

    def test_basic_history(self, provider_and_mock):
        provider, mock_create = provider_and_mock
        mock_create.return_value = make_mock_response(text="Sure, I can help.")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Help me."},
        ]
        result = provider.generate_with_history(messages)
        assert result["text"] == "Sure, I can help."

    def test_history_with_system_prompt(self, provider_and_mock):
        provider, mock_create = provider_and_mock
        messages = [{"role": "user", "content": "Hello"}]
        provider.generate_with_history(messages, system_prompt="Be helpful.")
        call_kwargs = mock_create.call_args[1]
        assert "Be helpful." in call_kwargs["system"]

    def test_history_json_mode(self, provider_and_mock):
        provider, mock_create = provider_and_mock
        messages = [{"role": "user", "content": "List"}]
        provider.generate_with_history(messages, json_mode=True)
        call_kwargs = mock_create.call_args[1]
        assert "Respond in valid JSON" in call_kwargs["system"]

    def test_history_system_and_json_mode(self, provider_and_mock):
        provider, mock_create = provider_and_mock
        messages = [{"role": "user", "content": "List"}]
        provider.generate_with_history(
            messages, system_prompt="Be brief.", json_mode=True
        )
        call_kwargs = mock_create.call_args[1]
        assert "Be brief." in call_kwargs["system"]
        assert "Respond in valid JSON" in call_kwargs["system"]

    def test_history_folds_system_role_messages(self, provider_and_mock):
        provider, mock_create = provider_and_mock
        messages = [
            {"role": "system", "content": "System context."},
            {"role": "user", "content": "Hello"},
        ]
        provider.generate_with_history(messages)
        call_kwargs = mock_create.call_args[1]
        # System role messages should be folded into system prompt
        assert "System context." in call_kwargs.get("system", "")
        # Sanitized messages should not include system role
        for msg in call_kwargs["messages"]:
            assert msg["role"] != "system"

    def test_history_token_usage(self, provider_and_mock):
        provider, mock_create = provider_and_mock
        mock_create.return_value = make_mock_response(
            input_tokens=200, output_tokens=80
        )
        messages = [{"role": "user", "content": "Hello"}]
        result = provider.generate_with_history(messages)
        assert result["input_tokens"] == 200
        assert result["output_tokens"] == 80

    def test_history_model_override(self, provider_and_mock):
        provider, mock_create = provider_and_mock
        messages = [{"role": "user", "content": "Hello"}]
        result = provider.generate_with_history(
            messages, model="claude-haiku-4-5-20251001"
        )
        assert result["model_id"] == "claude-haiku-4-5-20251001"


class TestListModels:
    """Tests for ClaudeProvider.list_models()."""

    def test_returns_list(self, provider):
        models = provider.list_models()
        assert isinstance(models, list)
        assert len(models) == 3

    def test_model_ids(self, provider):
        models = provider.list_models()
        ids = {m["model_id"] for m in models}
        assert "claude-opus-4-20250514" in ids
        assert "claude-sonnet-4-20250514" in ids
        assert "claude-haiku-4-5-20251001" in ids

    def test_model_has_required_keys(self, provider):
        models = provider.list_models()
        required_keys = {
            "model_id", "display_name", "context_window",
            "cost_per_1m_input", "cost_per_1m_output", "capabilities",
        }
        for model in models:
            assert required_keys.issubset(model.keys())

    def test_context_windows(self, provider):
        models = provider.list_models()
        for model in models:
            assert model["context_window"] == 200_000


class TestExtractText:
    """Tests for text extraction from response objects."""

    def test_single_block(self, provider):
        from aurarouter_claude.provider import ClaudeProvider
        response = MockMessage(content=[MockTextBlock(text="hello")])
        assert ClaudeProvider._extract_text(response) == "hello"

    def test_multiple_blocks(self, provider):
        from aurarouter_claude.provider import ClaudeProvider
        response = MockMessage(
            content=[MockTextBlock(text="hello "), MockTextBlock(text="world")]
        )
        assert ClaudeProvider._extract_text(response) == "hello world"

    def test_empty_content(self, provider):
        from aurarouter_claude.provider import ClaudeProvider
        response = MockMessage(content=[])
        assert ClaudeProvider._extract_text(response) == ""
