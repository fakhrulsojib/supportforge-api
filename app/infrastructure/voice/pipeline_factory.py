"""Voice pipeline factory — assembles per-session Pipecat pipelines.

Creates a configured RAG processor + STT/TTS adapters for a
specific tenant/user session with concurrency enforcement.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.core.exceptions import VoiceBusyError
from app.core.tenant_config import TenantVoiceConfig
from app.infrastructure.voice.pipecat_stt_adapter import PipecatSTTAdapter
from app.infrastructure.voice.pipecat_tts_adapter import PipecatTTSAdapter
from app.infrastructure.voice.rag_processor import SupportForgeRAGProcessor

logger = logging.getLogger(__name__)


class VoiceSessionManager:
    """Manages per-tenant voice session concurrency.

    Uses asyncio semaphores to enforce ``max_voice_sessions`` per tenant.
    Thread-safe for use with the async event loop.
    """

    def __init__(self, default_max_sessions: int = 3) -> None:
        self._default_max = default_max_sessions
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._active_counts: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, tenant_id: str, max_sessions: int | None = None) -> None:
        """Acquire a voice session slot for the given tenant.

        The availability check and semaphore acquire are performed
        atomically under a lock to prevent TOCTOU race conditions.

        Args:
            tenant_id: Tenant identifier.
            max_sessions: Override max sessions for this tenant.

        Raises:
            VoiceBusyError: If all session slots are occupied.
        """
        max_s = max_sessions or self._default_max

        async with self._lock:
            if tenant_id not in self._semaphores:
                self._semaphores[tenant_id] = asyncio.Semaphore(max_s)
                self._active_counts[tenant_id] = 0

            sem = self._semaphores[tenant_id]

            # Check availability while holding the lock (atomic)
            if sem.locked():
                raise VoiceBusyError()

            # Safe to acquire — we hold the lock so no other coroutine
            # can steal this slot between the check and the acquire.
            await sem.acquire()
            self._active_counts[tenant_id] = self._active_counts.get(tenant_id, 0) + 1

    async def release(self, tenant_id: str) -> None:
        """Release a voice session slot."""
        if tenant_id in self._semaphores:
            self._semaphores[tenant_id].release()
            async with self._lock:
                self._active_counts[tenant_id] = max(
                    0, self._active_counts.get(tenant_id, 0) - 1
                )

    def active_count(self, tenant_id: str) -> int:
        """Return current active session count for a tenant."""
        return self._active_counts.get(tenant_id, 0)


def create_voice_pipeline(
    chat_service: Any,
    stt_provider: Any,
    tts_provider: Any,
    tenant_id: str,
    user_id: str,
    voice_config: TenantVoiceConfig,
    tenant_model_config: Any,
    tenant_temperature: float = 0.7,
    tenant_blocklist: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble a per-session voice pipeline configuration.

    Returns a dict with:
        - ``rag_processor``: SupportForgeRAGProcessor instance
        - ``stt_adapter``: PipecatSTTAdapter instance
        - ``tts_adapter``: PipecatTTSAdapter instance

    The actual Pipecat Pipeline assembly (``Pipeline()``,
    ``SmallWebRTCTransport``, ``VADAnalyzer``) is done at the
    WebSocket endpoint level where the transport is available.
    """
    rag_processor = SupportForgeRAGProcessor(
        chat_service=chat_service,
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=None,
        tenant_model_config=tenant_model_config,
        tenant_temperature=tenant_temperature,
        tenant_blocklist=tenant_blocklist,
    )

    stt_adapter = PipecatSTTAdapter(stt_provider)
    tts_adapter = PipecatTTSAdapter(tts_provider, voice=voice_config.tts_voice)

    logger.info(
        "voice_pipeline_created",
        extra={
            "tenant_id": tenant_id,
            "user_id": user_id,
            "stt_provider": voice_config.stt_provider,
            "tts_provider": voice_config.tts_provider,
            "tts_voice": voice_config.tts_voice,
        },
    )

    return {
        "rag_processor": rag_processor,
        "stt_adapter": stt_adapter,
        "tts_adapter": tts_adapter,
    }
