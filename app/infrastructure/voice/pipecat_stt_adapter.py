"""Pipecat STT adapter — wraps our STTProvider for Pipecat pipeline.

Bridges the domain STTProvider ABC with Pipecat's STT service
interface. Handles errors gracefully to prevent WebSocket collapse.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PipecatSTTAdapter:
    """Adapter wrapping STTProvider for use in Pipecat pipelines.

    Used in the STT slot of the Pipecat pipeline. Delegates to our
    domain-layer STTProvider interface.
    """

    def __init__(self, stt_provider: Any) -> None:
        self._provider = stt_provider

    async def run_stt(self, audio: bytes, **kwargs: Any) -> str:
        """Transcribe audio via the wrapped STTProvider.

        Args:
            audio: Raw PCM Int16 mono audio bytes.

        Returns:
            Transcribed text, or empty string on error.
        """
        try:
            return await self._provider.transcribe(audio, **kwargs)
        except Exception:
            logger.exception("pipecat_stt_error")
            return ""
