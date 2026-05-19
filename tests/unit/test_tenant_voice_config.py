"""Tests for tenant voice configuration resolution.

Verifies the three-tier fallback: cloud → local → disabled.
"""

from __future__ import annotations

import pytest

from app.core.tenant_config import TenantVoiceConfig, resolve_tenant_voice_config


class TestTenantVoiceConfig:
    """Verify TenantVoiceConfig dataclass."""

    def test_defaults(self) -> None:
        """Default config has voice disabled."""
        config = TenantVoiceConfig()
        assert config.voice_enabled is False
        assert config.stt_provider is None
        assert config.tts_provider is None
        assert config.max_voice_sessions == 3

    def test_frozen(self) -> None:
        """TenantVoiceConfig is immutable."""
        config = TenantVoiceConfig()
        with pytest.raises(AttributeError):
            config.voice_enabled = True  # type: ignore[misc]


class TestResolveTenantVoiceConfig:
    """Verify three-tier resolution logic."""

    def test_none_config_returns_disabled(self) -> None:
        """None config returns disabled voice."""
        result = resolve_tenant_voice_config(None)
        assert result.voice_enabled is False

    def test_empty_config_returns_disabled(self) -> None:
        """Empty dict returns disabled voice."""
        result = resolve_tenant_voice_config({})
        assert result.voice_enabled is False

    def test_voice_disabled_explicitly(self) -> None:
        """voice_enabled=false in config → disabled."""
        result = resolve_tenant_voice_config({"voice_enabled": False})
        assert result.voice_enabled is False

    def test_cloud_stt_with_api_key(self) -> None:
        """Tier 1: cloud provider with API key → enabled."""
        result = resolve_tenant_voice_config(
            {
                "voice_enabled": True,
                "stt_provider": "deepgram",
                "stt_api_key": "test-key",
                "tts_provider": "elevenlabs",
                "tts_api_key": "test-key-2",
            },
        )
        assert result.voice_enabled is True
        assert result.stt_provider == "deepgram"
        assert result.tts_provider == "elevenlabs"

    def test_local_fallback_when_available(self) -> None:
        """Tier 2: no API key, local available → uses local."""
        result = resolve_tenant_voice_config(
            {"voice_enabled": True},
            stt_locally_available=True,
            tts_locally_available=True,
        )
        assert result.voice_enabled is True
        assert result.stt_provider == "whisper"
        assert result.tts_provider == "piper"

    def test_disabled_when_no_cloud_no_local(self) -> None:
        """Tier 3: no API key, no local → disabled."""
        result = resolve_tenant_voice_config(
            {"voice_enabled": True},
            stt_locally_available=False,
            tts_locally_available=False,
        )
        assert result.voice_enabled is False

    def test_partial_local_disables_voice(self) -> None:
        """Only STT local, no TTS → disabled (both required)."""
        result = resolve_tenant_voice_config(
            {"voice_enabled": True},
            stt_locally_available=True,
            tts_locally_available=False,
        )
        assert result.voice_enabled is False

    def test_custom_max_sessions(self) -> None:
        """max_voice_sessions is read from config."""
        result = resolve_tenant_voice_config(
            {"voice_enabled": True, "max_voice_sessions": 5},
            stt_locally_available=True,
            tts_locally_available=True,
        )
        assert result.max_voice_sessions == 5

    def test_invalid_max_sessions_clamped(self) -> None:
        """Negative max_voice_sessions clamped to 1."""
        result = resolve_tenant_voice_config(
            {"voice_enabled": True, "max_voice_sessions": -1},
            stt_locally_available=True,
            tts_locally_available=True,
        )
        assert result.max_voice_sessions >= 1

    def test_custom_tts_voice(self) -> None:
        """tts_voice is read from config."""
        result = resolve_tenant_voice_config(
            {
                "voice_enabled": True,
                "tts_voice": "en_GB-alan-medium",
            },
            stt_locally_available=True,
            tts_locally_available=True,
        )
        assert result.tts_voice == "en_GB-alan-medium"

    def test_never_raises(self) -> None:
        """resolve_tenant_voice_config never raises — returns disabled on error."""
        # Throw garbage in
        result = resolve_tenant_voice_config(
            {"voice_enabled": "not-a-bool", "max_voice_sessions": "abc"},
        )
        # Should not raise, just return defaults
        assert isinstance(result, TenantVoiceConfig)

    def test_encrypted_api_key_decrypt_failure_disables(self) -> None:
        """Invalid encrypted API key → logs warning, returns disabled."""
        result = resolve_tenant_voice_config(
            {
                "voice_enabled": True,
                "stt_provider": "deepgram",
                "stt_api_key": "invalid-ciphertext",
            },
            encryption_key="test-key",
            stt_locally_available=False,
            tts_locally_available=False,
        )
        # API key decryption failed, no local → disabled
        assert result.voice_enabled is False
