"""SupportForge RAG processor for the Pipecat pipeline.

This processor sits in the LLM slot of the Pipecat pipeline and
delegates to SupportForge's existing ChatService for all RAG/LLM
processing (retrieval → grading → generation → moderation).

No RAG logic is duplicated — the processor simply bridges Pipecat's
frame-based lifecycle with our service layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TextOutputFrame:
    """Lightweight text output frame (Pipecat-compatible)."""

    text: str


@dataclass
class EndOfResponseFrame:
    """Signals end of LLM response (Pipecat-compatible)."""

    pass


class SupportForgeRAGProcessor:
    """Pipecat processor that delegates to ChatService for RAG/LLM.

    Replaces Pipecat's default LLM service with SupportForge's
    existing RAG pipeline (retrieval → grading → generation → moderation).

    Pipeline position:
        Transport.input → VAD → STT → ContextAggregator → [THIS] → TTS → Transport.output
    """

    def __init__(
        self,
        chat_service: Any,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        tenant_model_config: Any,
        tenant_temperature: float,
        tenant_blocklist: list[str] | None,
        **kwargs: Any,
    ) -> None:
        self._chat_service = chat_service
        self._tenant_id = tenant_id
        self._user_id = user_id
        self._conversation_id = conversation_id
        self._model_config = tenant_model_config
        self._temperature = tenant_temperature
        self._blocklist = tenant_blocklist

        # Will be set by pipeline framework
        self.push_frame: Any = None

    async def process_transcript(self, transcript: str) -> None:
        """Process a transcribed text through the full RAG pipeline.

        Called when the STT service produces a transcript. Streams
        the RAG response as TextOutputFrame objects and signals
        completion with EndOfResponseFrame.

        Args:
            transcript: The transcribed text from STT.
        """
        if not transcript or not transcript.strip():
            return  # Silence — no-op

        try:
            async for chunk in self._chat_service.stream_message(
                message=transcript.strip(),
                tenant_id=self._tenant_id,
                user_id=self._user_id,
                conversation_id=self._conversation_id,
                temperature=self._temperature,
                tenant_blocklist=self._blocklist,
                tenant_chat_model=getattr(self._model_config, "chat_model", None),
                tenant_chat_provider=getattr(self._model_config, "chat_provider", None),
                tenant_embedding_model=getattr(self._model_config, "embedding_model", None),
                tenant_embedding_provider=getattr(self._model_config, "embedding_provider", None),
                tenant_gemini_api_key=getattr(self._model_config, "gemini_api_key", None),
                tenant_gemini_embedding_api_key=getattr(
                    self._model_config, "gemini_embedding_api_key", None
                ),
            ):
                if chunk["type"] == "token":
                    await self.push_frame(TextOutputFrame(text=chunk["data"]))
                elif chunk["type"] == "done":
                    # Track conversation ID for subsequent turns
                    done_data = chunk.get("data", {})
                    if isinstance(done_data, dict):
                        self._conversation_id = done_data.get(
                            "conversation_id", self._conversation_id
                        )
                    await self.push_frame(EndOfResponseFrame())

        except Exception:
            logger.exception("rag_processor_error")
            await self.push_frame(
                TextOutputFrame(
                    text="I'm sorry, I encountered an error processing your request."
                )
            )
            await self.push_frame(EndOfResponseFrame())
