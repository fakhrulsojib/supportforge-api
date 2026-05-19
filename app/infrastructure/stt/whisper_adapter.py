"""Whisper STT adapter — self-hosted speech-to-text via faster-whisper.

Uses ``faster-whisper`` for high-performance local transcription.
CPU-bound inference is offloaded to a thread pool via ``asyncio.to_thread``
to avoid blocking the async event loop.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from app.core.exceptions import STTError
from app.domain.interfaces.stt_provider import STTProvider

logger = structlog.get_logger(__name__)

# Default max audio size: 10 MB ≈ 5 minutes of PCM@16kHz mono
DEFAULT_MAX_AUDIO_BYTES = 10 * 1024 * 1024


class WhisperAdapter(STTProvider):
    """STT adapter using faster-whisper for local transcription.

    Args:
        model_size: Whisper model variant — ``tiny``, ``base``, ``small``,
            ``medium``, or ``large-v3``.
        device: Compute device — ``cpu`` (default) or ``cuda``.
        compute_type: Quantization — ``int8``, ``float16``, ``float32``.
        max_audio_bytes: Maximum audio buffer size in bytes.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        max_audio_bytes: int = DEFAULT_MAX_AUDIO_BYTES,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._max_audio_bytes = max_audio_bytes
        self._model: Any = None

    @property
    def provider_name(self) -> str:
        """Return ``'whisper'``."""
        return "whisper"

    def _load_model(self) -> None:
        """Load the faster-whisper model into memory.

        Separated for testability — tests can mock this method.
        """
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            logger.info(
                "whisper_model_loaded",
                model=self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        except ImportError:
            msg = "faster-whisper is not installed. Install with: pip install faster-whisper>=1.0"
            raise STTError(msg) from None

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
            sample_rate: Audio sample rate in Hz (default 16000).
            language: BCP-47 language code (default ``"en"``).

        Returns:
            Transcribed text string, stripped and joined.

        Raises:
            STTError: On empty input, oversized input, or model failure.
        """
        if not audio:
            raise STTError("Empty audio buffer")

        if len(audio) > self._max_audio_bytes:
            raise STTError(
                f"Audio exceeds {self._max_audio_bytes // 1024 // 1024}MB limit "
                f"(received {len(audio) // 1024 // 1024}MB)"
            )

        if self._model is None:
            raise STTError("Whisper model not loaded — call warm_up() first")

        try:
            import time

            import numpy as np  # type: ignore[import-untyped]

            logger.info(
                "whisper_transcribe_start",
                audio_bytes=len(audio),
                sample_rate=sample_rate,
                language=language,
                duration_secs=round(len(audio) / (sample_rate * 2), 2),
            )

            # Convert PCM Int16 to float32 normalized [-1.0, 1.0]
            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

            t0 = time.monotonic()
            # CPU-bound — offload to thread pool
            segments, info = await asyncio.to_thread(
                self._model.transcribe,
                samples,
                language=language,
            )

            text = " ".join(s.text.strip() for s in segments)
            elapsed = round(time.monotonic() - t0, 3)

            logger.info(
                "whisper_transcribe_done",
                text_length=len(text),
                text_preview=text[:80] if text else "(empty)",
                elapsed_secs=elapsed,
                language=getattr(info, "language", language),
                language_probability=round(getattr(info, "language_probability", 0), 3),
            )

            return text

        except STTError:
            raise
        except Exception as exc:
            msg = f"Transcription failed: {exc}"
            logger.exception("whisper_transcription_error")
            raise STTError(msg) from exc

    async def warm_up(self) -> None:
        """Pre-load the Whisper model into memory."""
        if self._model is None:
            logger.info("whisper_warm_up_start", model=self._model_size, device=self._device)
            await asyncio.to_thread(self._load_model)
            logger.info("whisper_warm_up_done")

    async def health_check(self) -> bool:
        """Return True if the model is loaded and ready."""
        ready = self._model is not None
        logger.debug("whisper_health_check", ready=ready)
        return ready
