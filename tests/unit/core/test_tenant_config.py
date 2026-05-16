"""Unit tests for tenant configuration helpers.

Covers:
    - resolve_tenant_models with None config
    - resolve_tenant_models with empty dict
    - resolve_tenant_models with valid chat_model only
    - resolve_tenant_models with valid embedding_model only
    - resolve_tenant_models with both models
    - Non-string model values ignored
    - Empty string model values ignored
    - Provider explicitly set to 'gemini'
    - Provider explicitly set to 'ollama'
    - Provider auto-detected from gemini model prefix
    - Provider defaults to None when no model set
    - Gemini API key decrypted from config
    - Gemini API key missing when provider is gemini
    - Invalid/non-string provider ignored
    - GEMINI_MODEL_PREFIXES constant correctness
"""

from __future__ import annotations

import pytest

from app.core.crypto import encrypt_value
from app.core.tenant_config import (
    CONFIG_CHAT_MODEL,
    CONFIG_CHAT_PROVIDER,
    CONFIG_EMBEDDING_MODEL,
    CONFIG_EMBEDDING_PROVIDER,
    CONFIG_GEMINI_API_KEY,
    CONFIG_GEMINI_EMBEDDING_API_KEY,
    GEMINI_MODEL_PREFIXES,
    TenantModelConfig,
    resolve_tenant_models,
)


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def secret_key() -> str:
    """Application secret key for encryption tests."""
    return "test-secret-key-for-unit-tests"


@pytest.fixture
def sample_api_key() -> str:
    """A realistic Gemini API key."""
    return "AIzaSyD-fake-test-key-1234567890abcdef"


# ── None / Empty Config ─────────────────────────────────────────


class TestEmptyConfig:
    """Tests for None and empty config_json."""

    def test_none_config(self) -> None:
        """None config should return all-None defaults."""
        result = resolve_tenant_models(None)
        assert result.chat_model is None
        assert result.embedding_model is None
        assert result.chat_provider is None
        assert result.gemini_api_key is None

    def test_empty_dict(self) -> None:
        """Empty dict should return all-None defaults."""
        result = resolve_tenant_models({})
        assert result.chat_model is None
        assert result.embedding_model is None
        assert result.chat_provider is None
        assert result.gemini_api_key is None


# ── Basic Model Selection (backward compat) ─────────────────────


class TestBasicModelSelection:
    """Tests for chat_model and embedding_model selection."""

    def test_chat_model_only(self) -> None:
        """Config with only chat_model set."""
        result = resolve_tenant_models({CONFIG_CHAT_MODEL: "gemma3:4b"})
        assert result.chat_model == "gemma3:4b"
        assert result.embedding_model is None

    def test_embedding_model_only(self) -> None:
        """Config with only embedding_model set."""
        result = resolve_tenant_models({CONFIG_EMBEDDING_MODEL: "nomic-embed-text"})
        assert result.embedding_model == "nomic-embed-text"
        assert result.chat_model is None

    def test_both_models(self) -> None:
        """Config with both models set."""
        result = resolve_tenant_models({
            CONFIG_CHAT_MODEL: "gemma3:4b",
            CONFIG_EMBEDDING_MODEL: "nomic-embed-text",
        })
        assert result.chat_model == "gemma3:4b"
        assert result.embedding_model == "nomic-embed-text"

    def test_non_string_model_ignored(self) -> None:
        """Non-string model values should be ignored."""
        result = resolve_tenant_models({CONFIG_CHAT_MODEL: 42})
        assert result.chat_model is None

    def test_empty_string_model_ignored(self) -> None:
        """Empty string model values should be ignored."""
        result = resolve_tenant_models({CONFIG_CHAT_MODEL: ""})
        assert result.chat_model is None


# ── Provider Selection ──────────────────────────────────────────


class TestProviderSelection:
    """Tests for chat_provider resolution."""

    def test_explicit_gemini_provider(self) -> None:
        """Provider explicitly set to 'gemini'."""
        result = resolve_tenant_models({
            CONFIG_CHAT_PROVIDER: "gemini",
            CONFIG_CHAT_MODEL: "gemini-2.5-flash",
        })
        assert result.chat_provider == "gemini"

    def test_explicit_ollama_provider(self) -> None:
        """Provider explicitly set to 'ollama'."""
        result = resolve_tenant_models({
            CONFIG_CHAT_PROVIDER: "ollama",
            CONFIG_CHAT_MODEL: "gemma3:4b",
        })
        assert result.chat_provider == "ollama"

    def test_auto_detect_gemini_from_model_name(self) -> None:
        """Provider auto-detected when model starts with 'gemini-'."""
        result = resolve_tenant_models({
            CONFIG_CHAT_MODEL: "gemini-2.5-flash-lite",
        })
        assert result.chat_provider == "gemini"

    def test_auto_detect_ollama_from_model_name(self) -> None:
        """Non-gemini model names default to None (caller falls back to Ollama)."""
        result = resolve_tenant_models({
            CONFIG_CHAT_MODEL: "phi4-mini",
        })
        assert result.chat_provider is None

    def test_no_provider_no_model(self) -> None:
        """No provider when no model selected."""
        result = resolve_tenant_models({})
        assert result.chat_provider is None

    def test_non_string_provider_ignored(self) -> None:
        """Non-string provider values should be ignored."""
        result = resolve_tenant_models({CONFIG_CHAT_PROVIDER: 123})
        assert result.chat_provider is None


# ── Gemini API Key ──────────────────────────────────────────────


class TestGeminiApiKey:
    """Tests for encrypted API key extraction."""

    def test_api_key_decrypted(
        self, secret_key: str, sample_api_key: str
    ) -> None:
        """Encrypted API key in config should be decrypted."""
        encrypted = encrypt_value(sample_api_key, secret_key)
        result = resolve_tenant_models(
            {CONFIG_GEMINI_API_KEY: encrypted},
            encryption_key=secret_key,
        )
        assert result.gemini_api_key == sample_api_key

    def test_api_key_missing(self) -> None:
        """Missing API key should return None."""
        result = resolve_tenant_models({CONFIG_CHAT_PROVIDER: "gemini"})
        assert result.gemini_api_key is None

    def test_api_key_decryption_failure_returns_none(self) -> None:
        """Invalid encrypted value should return None (not crash)."""
        result = resolve_tenant_models(
            {CONFIG_GEMINI_API_KEY: "not-valid-ciphertext"},
            encryption_key="some-key",
        )
        assert result.gemini_api_key is None

    def test_api_key_without_encryption_key(self) -> None:
        """No encryption_key param → key not decrypted, returns None."""
        result = resolve_tenant_models(
            {CONFIG_GEMINI_API_KEY: "some-encrypted-value"},
        )
        assert result.gemini_api_key is None


# ── Constants ───────────────────────────────────────────────────


class TestConstants:
    """Tests for module constants."""

    def test_gemini_prefix_tuple(self) -> None:
        """GEMINI_MODEL_PREFIXES should be a tuple of string prefixes."""
        assert isinstance(GEMINI_MODEL_PREFIXES, tuple)
        assert all(isinstance(p, str) for p in GEMINI_MODEL_PREFIXES)
        assert "gemini-" in GEMINI_MODEL_PREFIXES

    def test_config_key_strings(self) -> None:
        """Config keys should be non-empty strings."""
        assert isinstance(CONFIG_CHAT_PROVIDER, str) and CONFIG_CHAT_PROVIDER
        assert isinstance(CONFIG_GEMINI_API_KEY, str) and CONFIG_GEMINI_API_KEY

    def test_dataclass_frozen(self) -> None:
        """TenantModelConfig should be frozen (immutable)."""
        config = TenantModelConfig()
        with pytest.raises(AttributeError):
            config.chat_model = "test"  # type: ignore[misc]


# ── Embedding Provider ──────────────────────────────────────────


class TestEmbeddingProvider:
    """Tests for embedding_provider resolution."""

    def test_explicit_gemini_embedding_provider(self) -> None:
        """Embedding provider explicitly set to 'gemini'."""
        result = resolve_tenant_models({
            CONFIG_EMBEDDING_PROVIDER: "gemini",
            CONFIG_EMBEDDING_MODEL: "gemini-embedding-2",
        })
        assert result.embedding_provider == "gemini"

    def test_explicit_ollama_embedding_provider(self) -> None:
        """Embedding provider explicitly set to 'ollama'."""
        result = resolve_tenant_models({
            CONFIG_EMBEDDING_PROVIDER: "ollama",
            CONFIG_EMBEDDING_MODEL: "nomic-embed-text",
        })
        assert result.embedding_provider == "ollama"

    def test_auto_detect_gemini_embedding_from_model_name(self) -> None:
        """Provider auto-detected when embedding model starts with 'gemini-'."""
        result = resolve_tenant_models({
            CONFIG_EMBEDDING_MODEL: "gemini-embedding-2",
        })
        assert result.embedding_provider == "gemini"

    def test_auto_detect_ollama_embedding_from_model_name(self) -> None:
        """Non-gemini embedding model defaults to None."""
        result = resolve_tenant_models({
            CONFIG_EMBEDDING_MODEL: "nomic-embed-text",
        })
        assert result.embedding_provider is None

    def test_no_embedding_provider_no_model(self) -> None:
        """No embedding provider when no embedding model selected."""
        result = resolve_tenant_models({})
        assert result.embedding_provider is None


# ── Gemini Embedding API Key ────────────────────────────────────


class TestGeminiEmbeddingApiKey:
    """Tests for separate encrypted embedding API key extraction."""

    def test_embedding_key_decrypted(
        self, secret_key: str, sample_api_key: str
    ) -> None:
        """Encrypted embedding API key in config should be decrypted."""
        encrypted = encrypt_value(sample_api_key, secret_key)
        result = resolve_tenant_models(
            {CONFIG_GEMINI_EMBEDDING_API_KEY: encrypted},
            encryption_key=secret_key,
        )
        assert result.gemini_embedding_api_key == sample_api_key

    def test_embedding_key_independent_from_chat_key(
        self, secret_key: str,
    ) -> None:
        """Chat key and embedding key should be independent."""
        chat_key = "AIza-chat-key-1234"
        embed_key = "AIza-embed-key-5678"
        encrypted_chat = encrypt_value(chat_key, secret_key)
        encrypted_embed = encrypt_value(embed_key, secret_key)
        result = resolve_tenant_models(
            {
                CONFIG_GEMINI_API_KEY: encrypted_chat,
                CONFIG_GEMINI_EMBEDDING_API_KEY: encrypted_embed,
            },
            encryption_key=secret_key,
        )
        assert result.gemini_api_key == chat_key
        assert result.gemini_embedding_api_key == embed_key

    def test_embedding_key_missing(self) -> None:
        """Missing embedding API key should return None."""
        result = resolve_tenant_models({CONFIG_EMBEDDING_PROVIDER: "gemini"})
        assert result.gemini_embedding_api_key is None

    def test_embedding_key_decryption_failure_returns_none(self) -> None:
        """Invalid encrypted value should return None (not crash)."""
        result = resolve_tenant_models(
            {CONFIG_GEMINI_EMBEDDING_API_KEY: "not-valid-ciphertext"},
            encryption_key="some-key",
        )
        assert result.gemini_embedding_api_key is None
