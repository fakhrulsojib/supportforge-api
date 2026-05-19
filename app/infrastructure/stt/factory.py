"""STT provider factory — config-driven with lazy imports.

Follows the same pattern as ``app.infrastructure.llm.factory``.
Each branch only imports its adapter when selected, so missing
optional dependencies (e.g. ``faster-whisper``) only fail for
the affected provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

        return WhisperAdapter(**kwargs)  # accepts model_size, device, compute_type, max_audio_bytes

    if provider == "azure":
        from app.infrastructure.stt.azure_adapter import AzureSTTAdapter

        return AzureSTTAdapter(**kwargs)  # accepts subscription_key, region

    if provider == "deepgram":
        # Placeholder — deepgram adapter is in voice-cloud extras
        msg = "Deepgram STT adapter not available. Install with: pip install 'supportforge[voice-cloud]'"
        raise ValueError(msg)

    if provider == "google":
        # Placeholder — Google STT adapter is in voice-cloud extras
        msg = "Google STT adapter not available. Install with: pip install 'supportforge[voice-cloud]'"
        raise ValueError(msg)

    msg = f"Unknown STT provider: '{provider}'. Available: whisper, azure, deepgram, google"
    raise ValueError(msg)
