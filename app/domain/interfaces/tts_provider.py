"""Text-to-Speech provider interface (domain port).

Defines the ABC that all TTS adapters must implement.
Pure domain — ZERO framework imports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class TTSProvider(ABC):
    """Port for Text-to-Speech synthesis.

    Implementations must handle audio synthesis (single + streaming),
    voice listing, model warm-up, and health checking.
    All I/O methods are async.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier (e.g. 'piper', 'elevenlabs')."""
        ...

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        *,
        sample_rate: int = 22050,
    ) -> bytes:
        """Synthesize text to raw PCM Int16 mono audio bytes.

        Args:
            text: Text to synthesize.
            voice: Provider-specific voice identifier.
            sample_rate: Output sample rate in Hz.

        Returns:
            Raw PCM Int16 mono audio bytes (no WAV header).

        Raises:
            TTSError: On synthesis failure.
        """
        ...

    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        voice: str = "default",
        *,
        sample_rate: int = 22050,
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks for sentence-level playback.

        Yields:
            PCM Int16 mono audio byte chunks.
        """
        ...
        yield b""  # pragma: no cover — makes this a generator

    @abstractmethod
    async def list_voices(self) -> list[dict[str, str]]:
        """List available voices.

        Returns:
            List of dicts with at minimum ``id`` and ``name`` keys.
        """
        ...

    @abstractmethod
    async def warm_up(self) -> None:
        """Pre-load voice model into memory.

        Called once at startup. Cloud providers should no-op.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the TTS service is operational."""
        ...
