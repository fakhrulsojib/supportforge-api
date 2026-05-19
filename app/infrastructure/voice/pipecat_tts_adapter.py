"""Pipecat TTS adapter — wraps our TTSProvider for Pipecat pipeline.

Bridges the domain TTSProvider ABC with Pipecat's TTS service
interface. Handles errors gracefully to prevent WebSocket collapse.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PipecatTTSAdapter:
    """Adapter wrapping TTSProvider for use in Pipecat pipelines.

    Used in the TTS slot of the Pipecat pipeline. Delegates to our
    domain-layer TTSProvider interface.
    """

    def __init__(self, tts_provider: Any, voice: str = "default") -> None:
        self._provider = tts_provider
        self._voice = voice

    async def run_tts(self, text: str, **kwargs: Any) -> bytes:
        """Synthesize text via the wrapped TTSProvider.

        Args:
            text: Text to synthesize.

        Returns:
            Raw PCM Int16 mono audio bytes, or empty bytes on error.
        """
        try:
            return await self._provider.synthesize(text, voice=self._voice, **kwargs)
        except Exception:
            logger.exception("pipecat_tts_error")
            return b""
