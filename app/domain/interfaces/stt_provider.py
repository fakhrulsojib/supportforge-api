"""Speech-to-Text provider interface (domain port).

Defines the ABC that all STT adapters must implement.
Pure domain — ZERO framework imports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class STTProvider(ABC):
    """Port for Speech-to-Text communication.

    Implementations must handle audio transcription, model warm-up,
    and health checking.  All I/O methods are async.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier (e.g. 'whisper', 'deepgram')."""
        ...

    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        *,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> str:
        """Transcribe raw PCM Int16 mono audio to text.

        Args:
            audio: Raw PCM Int16 mono audio bytes.
            sample_rate: Audio sample rate in Hz.
            language: BCP-47 language code.

        Returns:
            Transcribed text string.

        Raises:
            STTError: On transcription failure.
        """
        ...

    @abstractmethod
    async def warm_up(self) -> None:
        """Pre-load models into memory.

        Called once at startup to avoid cold-start latency on the
        first voice request.  Cloud provider implementations should
        no-op this method.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the STT service is operational."""
        ...
