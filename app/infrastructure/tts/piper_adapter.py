"""Piper TTS adapter — self-hosted text-to-speech via piper-tts.

Uses ``piper-tts`` for local synthesis. CPU-bound inference is offloaded
to a thread pool via ``asyncio.to_thread``.

Requires the ``espeak-ng`` system package (documented in Dockerfile).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from app.core.exceptions import TTSError
from app.domain.interfaces.tts_provider import TTSProvider

logger = logging.getLogger(__name__)


class PiperAdapter(TTSProvider):
    """TTS adapter using Piper for local speech synthesis.

    Args:
        voice: Piper voice model name (e.g. ``en_US-lessac-medium``)
            or absolute path to a ``.onnx`` model file.
        models_dir: Directory to search for model files. Defaults to
            ``<project_root>/.voice_models/``.
    """

    def __init__(
        self,
        voice: str = "en_US-lessac-medium",
        models_dir: str | None = None,
    ) -> None:
        self._voice_name = voice
        self._models_dir = models_dir
        self._voice_model: Any = None

    @property
    def provider_name(self) -> str:
        """Return ``'piper'``."""
        return "piper"

    def _resolve_model_path(self) -> str:
        """Resolve the voice name to an actual .onnx file path.

        Resolution order:
        1. If ``voice`` is an absolute path and exists → use it directly.
        2. Look for ``<voice>.onnx`` in ``models_dir``.
        3. Look for ``<voice>.onnx`` in ``<project_root>/.voice_models/``.
        4. Fall back to the raw name (let Piper raise its own error).
        """
        from pathlib import Path

        # 1. Absolute path
        if Path(self._voice_name).is_absolute() and Path(self._voice_name).exists():
            return self._voice_name

        # 2. Explicit models dir
        if self._models_dir:
            candidate = Path(self._models_dir) / f"{self._voice_name}.onnx"
            if candidate.exists():
                return str(candidate)

        # 3. Default project .voice_models/
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        candidate = project_root / ".voice_models" / f"{self._voice_name}.onnx"
        if candidate.exists():
            return str(candidate)

        # 4. Raw fallback
        return self._voice_name

    def _load_voice(self) -> None:
        """Load the Piper voice model.

        Separated for testability — tests can mock this method.
        """
        try:
            from piper import PiperVoice  # type: ignore[import-untyped]

            model_path = self._resolve_model_path()
            self._voice_model = PiperVoice.load(model_path)
            logger.info("piper_voice_loaded", extra={"voice": self._voice_name, "path": model_path})
        except ImportError:
            msg = "piper-tts is not installed. Install with: pip install piper-tts>=1.4"
            raise TTSError(msg) from None

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
            voice: Ignored (uses instance voice).
            sample_rate: Output sample rate.

        Returns:
            Raw PCM Int16 mono audio bytes. Empty bytes for empty input.

        Raises:
            TTSError: On synthesis failure.
        """
        if not text or not text.strip():
            return b""

        if self._voice_model is None:
            raise TTSError("Piper voice model not loaded — call warm_up() first")

        try:
            def _sync_synthesize() -> bytes:
                """Run synthesis synchronously (CPU-bound)."""
                chunks: list[bytes] = []
                for audio_chunk in self._voice_model.synthesize(text):
                    chunks.append(audio_chunk)
                return b"".join(chunks)

            return await asyncio.to_thread(_sync_synthesize)

        except TTSError:
            raise
        except Exception as exc:
            msg = f"Synthesis failed: {exc}"
            logger.exception("piper_synthesis_error")
            raise TTSError(msg) from exc

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
        if not text or not text.strip():
            return

        if self._voice_model is None:
            raise TTSError("Piper voice model not loaded — call warm_up() first")

        try:
            # Split on sentence boundaries (after .!? followed by whitespace)
            # Avoids breaking on abbreviations ("Dr."), decimals ("$3.99"), etc.
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            for sentence in sentences:
                audio = await self.synthesize(sentence, voice, sample_rate=sample_rate)
                if audio:
                    yield audio
        except TTSError:
            raise
        except Exception as exc:
            msg = f"Stream synthesis failed: {exc}"
            logger.exception("piper_stream_error")
            raise TTSError(msg) from exc

    async def list_voices(self) -> list[dict[str, str]]:
        """List available Piper voices.

        Returns:
            List of dicts with ``id`` and ``name`` keys.
        """
        # For now, return the configured voice
        return [
            {
                "id": self._voice_name,
                "name": self._voice_name.replace("-", " ").replace("_", " ").title(),
            },
        ]

    async def warm_up(self) -> None:
        """Pre-load the Piper voice model.

        Offloaded to a thread pool to avoid blocking the event loop
        during model loading.
        """
        if self._voice_model is None:
            await asyncio.to_thread(self._load_voice)

    async def health_check(self) -> bool:
        """Return True if the voice model is loaded."""
        return self._voice_model is not None
