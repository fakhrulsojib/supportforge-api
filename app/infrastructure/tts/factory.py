"""TTS provider factory — config-driven with lazy imports.

Follows the same pattern as ``app.infrastructure.stt.factory``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
        msg = "Google TTS adapter not available. Install with: pip install 'supportforge[voice-cloud]'"
        raise ValueError(msg)

    if provider == "elevenlabs":
        msg = "ElevenLabs TTS adapter not available. Install with: pip install 'supportforge[voice-cloud]'"
        raise ValueError(msg)

    msg = f"Unknown TTS provider: '{provider}'. Available: piper, google, elevenlabs"
    raise ValueError(msg)
