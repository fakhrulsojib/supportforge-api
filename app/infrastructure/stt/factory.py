"""STT provider factory — config-driven with lazy imports.

Follows the same pattern as ``app.infrastructure.llm.factory``.
Each branch only imports its adapter when selected, so missing
optional dependencies (e.g. ``faster-whisper``) only fail for
the affected provider.
"""

from __future__ import annotations

from typing import Any

from app.domain.interfaces.stt_provider import STTProvider


def get_stt_provider(provider: str, **kwargs: Any) -> STTProvider:
    """Create an STT provider by name.

    Args:
        provider: One of ``"whisper"``, ``"deepgram"``, ``"google"``.
        **kwargs: Provider-specific configuration.

    Returns:
        An initialized STTProvider instance.

    Raises:
        ValueError: If the provider name is not recognized.
    """
    if provider == "whisper":
        from app.infrastructure.stt.whisper_adapter import WhisperAdapter

        return WhisperAdapter(**kwargs)

    if provider == "deepgram":
        # Placeholder — deepgram adapter is in voice-cloud extras
        msg = "Deepgram STT adapter requires 'deepgram-sdk'. Install with: pip install deepgram-sdk>=3.0"
        raise ImportError(msg)

    if provider == "google":
        # Placeholder — Google STT adapter is in voice-cloud extras
        msg = "Google STT adapter requires 'google-cloud-speech'. Install with: pip install google-cloud-speech>=2.0"
        raise ImportError(msg)

    msg = f"Unknown STT provider: '{provider}'. Available: whisper, deepgram, google"
    raise ValueError(msg)
