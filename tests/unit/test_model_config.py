#!/usr/bin/env python
# coding: utf-8
"""Unit tests for model provider configuration."""

import pytest

from cs_copilot.model_config import (
    DEFAULT_OPENROUTER_MAX_TOKENS,
    load_model_from_config,
)


@pytest.fixture(autouse=True)
def clear_model_env(monkeypatch):
    """Keep model_config tests independent from the developer shell."""
    for key in (
        "MODEL_PROVIDER",
        "MODEL_ID",
        "MODEL_MAX_TOKENS",
        "MODELCONF_PATH",
        "OLLAMA_HOST",
    ):
        monkeypatch.delenv(key, raising=False)


def _write_modelconf(tmp_path, content: str) -> str:
    path = tmp_path / ".modelconf"
    path.write_text(content)
    return str(path)


def test_openrouter_deepseek_uses_large_default_token_budget(tmp_path):
    config_path = _write_modelconf(
        tmp_path,
        "provider=openrouter\nmodel_id=deepseek/deepseek-v3.2\n",
    )

    model = load_model_from_config(config_path)

    assert model.id == "deepseek/deepseek-v3.2"
    assert model.max_tokens == DEFAULT_OPENROUTER_MAX_TOKENS
    assert model.supports_native_structured_outputs is False


def test_openrouter_max_tokens_can_be_configured(tmp_path):
    config_path = _write_modelconf(
        tmp_path,
        "provider=openrouter\nmodel_id=openai/gpt-5-mini\nmax_tokens=4096\n",
    )

    model = load_model_from_config(config_path)

    assert model.max_tokens == 4096
    assert model.supports_native_structured_outputs is True


def test_model_max_tokens_env_overrides_modelconf(tmp_path, monkeypatch):
    config_path = _write_modelconf(
        tmp_path,
        "provider=openrouter\nmodel_id=deepseek/deepseek-v3.2\nmax_tokens=4096\n",
    )
    monkeypatch.setenv("MODEL_MAX_TOKENS", "16384")

    model = load_model_from_config(config_path)

    assert model.max_tokens == 16384
    assert model.supports_native_structured_outputs is False


def test_openrouter_rejects_invalid_max_tokens(tmp_path):
    config_path = _write_modelconf(
        tmp_path,
        "provider=openrouter\nmodel_id=deepseek/deepseek-v3.2\nmax_tokens=0\n",
    )

    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        load_model_from_config(config_path)


def test_native_deepseek_provider_is_unchanged(tmp_path):
    config_path = _write_modelconf(
        tmp_path,
        "provider=deepseek\nmodel_id=deepseek-chat\n",
    )

    model = load_model_from_config(config_path)

    assert model.id == "deepseek-chat"
    assert model.max_tokens is None
    assert model.supports_native_structured_outputs is False
