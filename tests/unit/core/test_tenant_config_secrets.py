"""Unit tests for secrets-priority resolution in tenant config.

Tests the new `secrets` parameter in `resolve_tenant_models()` which
enables reading Gemini API keys from the `tenant_secrets` table with
fallback to encrypted config_json values.

Covers:
    - Secrets take priority over config_json for API keys
    - config_json fallback when secrets dict is empty/None
    - Mixed: secrets has chat key, config_json has embedding key
    - Empty string secrets are ignored (fall through to config_json)
    - Both sources present — secrets wins
    - Secret key names match expected constants
"""

from __future__ import annotations

import pytest

from app.core.crypto import encrypt_value
from app.core.tenant_config import (
    CONFIG_GEMINI_API_KEY,
    CONFIG_GEMINI_EMBEDDING_API_KEY,
    SECRET_GEMINI_API_KEY,
    SECRET_GEMINI_EMBEDDING_API_KEY,
    resolve_tenant_models,
)


@pytest.fixture
def secret_key() -> str:
    """Application secret key for encryption tests."""
    return "test-secret-key-for-unit-tests"


@pytest.fixture
def sample_chat_key() -> str:
    return "AIzaSyD-chat-key-1234567890abcdef"


@pytest.fixture
def sample_embed_key() -> str:
    return "AIzaSyD-embed-key-0987654321fedcba"


class TestSecretsConstants:
    """Verify secret key name constants are correctly defined."""

    def test_secret_gemini_api_key(self) -> None:
        assert SECRET_GEMINI_API_KEY == "gemini_api_key"

    def test_secret_gemini_embedding_api_key(self) -> None:
        assert SECRET_GEMINI_EMBEDDING_API_KEY == "gemini_embedding_api_key"


class TestSecretsPriority:
    """Tests for secrets > config_json priority resolution."""

    def test_secrets_takes_priority_over_config_json(
        self, secret_key: str, sample_chat_key: str
    ) -> None:
        """When both sources have a key, secrets dict wins."""
        encrypted = encrypt_value("old-key-in-config", secret_key)
        result = resolve_tenant_models(
            {CONFIG_GEMINI_API_KEY: encrypted},
            encryption_key=secret_key,
            secrets={SECRET_GEMINI_API_KEY: sample_chat_key},
        )
        assert result.gemini_api_key == sample_chat_key

    def test_embedding_key_from_secrets(
        self, secret_key: str, sample_embed_key: str
    ) -> None:
        """Embedding key resolved from secrets dict."""
        result = resolve_tenant_models(
            {},
            encryption_key=secret_key,
            secrets={SECRET_GEMINI_EMBEDDING_API_KEY: sample_embed_key},
        )
        assert result.gemini_embedding_api_key == sample_embed_key

    def test_both_keys_from_secrets(
        self, sample_chat_key: str, sample_embed_key: str
    ) -> None:
        """Both chat and embedding keys from secrets."""
        result = resolve_tenant_models(
            {},
            secrets={
                SECRET_GEMINI_API_KEY: sample_chat_key,
                SECRET_GEMINI_EMBEDDING_API_KEY: sample_embed_key,
            },
        )
        assert result.gemini_api_key == sample_chat_key
        assert result.gemini_embedding_api_key == sample_embed_key


class TestSecretsConfigJsonFallback:
    """Tests for fallback to config_json when secrets are absent."""

    def test_fallback_to_config_json_when_secrets_none(
        self, secret_key: str, sample_chat_key: str
    ) -> None:
        """No secrets dict → fall back to config_json."""
        encrypted = encrypt_value(sample_chat_key, secret_key)
        result = resolve_tenant_models(
            {CONFIG_GEMINI_API_KEY: encrypted},
            encryption_key=secret_key,
            secrets=None,
        )
        assert result.gemini_api_key == sample_chat_key

    def test_fallback_to_config_json_when_secrets_empty(
        self, secret_key: str, sample_chat_key: str
    ) -> None:
        """Empty secrets dict → fall back to config_json."""
        encrypted = encrypt_value(sample_chat_key, secret_key)
        result = resolve_tenant_models(
            {CONFIG_GEMINI_API_KEY: encrypted},
            encryption_key=secret_key,
            secrets={},
        )
        assert result.gemini_api_key == sample_chat_key

    def test_fallback_only_for_missing_key(
        self, secret_key: str, sample_chat_key: str, sample_embed_key: str
    ) -> None:
        """Mixed: secrets has chat key, config_json has embedding key."""
        encrypted_embed = encrypt_value(sample_embed_key, secret_key)
        result = resolve_tenant_models(
            {CONFIG_GEMINI_EMBEDDING_API_KEY: encrypted_embed},
            encryption_key=secret_key,
            secrets={SECRET_GEMINI_API_KEY: sample_chat_key},
        )
        assert result.gemini_api_key == sample_chat_key  # from secrets
        assert result.gemini_embedding_api_key == sample_embed_key  # from config_json

    def test_no_keys_anywhere(self) -> None:
        """No keys in either source → both None."""
        result = resolve_tenant_models(
            {},
            secrets={},
        )
        assert result.gemini_api_key is None
        assert result.gemini_embedding_api_key is None

    def test_none_config_none_secrets(self) -> None:
        """Both None → defaults."""
        result = resolve_tenant_models(None, secrets=None)
        assert result.gemini_api_key is None
        assert result.gemini_embedding_api_key is None


class TestSecretsDoNotAffectNonKeyFields:
    """Verify secrets parameter only affects API key resolution."""

    def test_model_selection_unaffected(self, sample_chat_key: str) -> None:
        """Model selection from config_json is not changed by secrets."""
        result = resolve_tenant_models(
            {"chat_model": "gemma3:4b", "chat_provider": "ollama"},
            secrets={SECRET_GEMINI_API_KEY: sample_chat_key},
        )
        assert result.chat_model == "gemma3:4b"
        assert result.chat_provider == "ollama"
        assert result.gemini_api_key == sample_chat_key

    def test_embedding_model_unaffected(self) -> None:
        """Embedding model from config_json is not changed by secrets."""
        result = resolve_tenant_models(
            {"embedding_model": "nomic-embed-text"},
            secrets={},
        )
        assert result.embedding_model == "nomic-embed-text"
