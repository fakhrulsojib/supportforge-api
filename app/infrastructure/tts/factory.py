"""TTS provider factory — config-driven with lazy imports.

Follows the same pattern as ``app.infrastructure.stt.factory``.
"""

from __future__ import annotations

from typing import Any

from app.domain.interfaces.tts_provider import TTSProvider


def get_tts_provider(provider: str, **kwargs: Any) -> TTSProvider:
    """Create a TTS provider by name.

    Args:
        provider: One of ``"piper"``, ``"google"``, ``"elevenlabs"``.
        **kwargs: Provider-specific configuration.

    Returns:
        An initialized TTSProvider instance.

    Raises:
        ValueError: If the provider name is not recognized.
    """
    if provider == "piper":
        from app.infrastructure.tts.piper_adapter import PiperAdapter

        return PiperAdapter(**kwargs)

    if provider == "google":
        msg = "Google TTS adapter requires 'google-cloud-texttospeech'. Install with: pip install google-cloud-texttospeech>=2.0"
        raise ImportError(msg)

    if provider == "elevenlabs":
        msg = "ElevenLabs TTS adapter requires 'elevenlabs'. Install with: pip install elevenlabs>=1.0"
        raise ImportError(msg)

    msg = f"Unknown TTS provider: '{provider}'. Available: piper, google, elevenlabs"
    raise ValueError(msg)
