"""Tests for VoiceSessionManager and pipeline factory.

Verifies concurrency enforcement and pipeline assembly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.exceptions import VoiceBusyError
from app.core.tenant_config import TenantVoiceConfig
from app.infrastructure.voice.pipeline_factory import (
    VoiceSessionManager,
    create_voice_pipeline,
)


class TestVoiceSessionManager:
    """Verify per-tenant concurrency enforcement."""

    @pytest.mark.asyncio
    async def test_acquire_and_release(self) -> None:
        """Basic acquire/release cycle."""
        mgr = VoiceSessionManager(default_max_sessions=2)
        await mgr.acquire("t1")
        assert mgr.active_count("t1") == 1
        await mgr.release("t1")
        assert mgr.active_count("t1") == 0

    @pytest.mark.asyncio
    async def test_acquire_up_to_max(self) -> None:
        """Can acquire up to max_sessions slots."""
        mgr = VoiceSessionManager(default_max_sessions=2)
        await mgr.acquire("t1")
        await mgr.acquire("t1")
        assert mgr.active_count("t1") == 2

    @pytest.mark.asyncio
    async def test_exceed_max_raises_voice_busy(self) -> None:
        """Exceeding max_sessions raises VoiceBusyError."""
        mgr = VoiceSessionManager(default_max_sessions=1)
        await mgr.acquire("t1")
        with pytest.raises(VoiceBusyError):
            await mgr.acquire("t1")

    @pytest.mark.asyncio
    async def test_different_tenants_independent(self) -> None:
        """Different tenants have independent session pools."""
        mgr = VoiceSessionManager(default_max_sessions=1)
        await mgr.acquire("t1")
        await mgr.acquire("t2")  # Should not raise
        assert mgr.active_count("t1") == 1
        assert mgr.active_count("t2") == 1

    @pytest.mark.asyncio
    async def test_release_allows_reacquire(self) -> None:
        """Releasing a slot allows a new acquire."""
        mgr = VoiceSessionManager(default_max_sessions=1)
        await mgr.acquire("t1")
        await mgr.release("t1")
        await mgr.acquire("t1")  # Should not raise

    @pytest.mark.asyncio
    async def test_custom_max_sessions(self) -> None:
        """Custom max_sessions per acquire call."""
        mgr = VoiceSessionManager(default_max_sessions=1)
        await mgr.acquire("t1", max_sessions=3)
        await mgr.acquire("t1", max_sessions=3)
        await mgr.acquire("t1", max_sessions=3)
        assert mgr.active_count("t1") == 3

    @pytest.mark.asyncio
    async def test_release_unknown_tenant_noop(self) -> None:
        """Releasing a non-acquired tenant is a no-op."""
        mgr = VoiceSessionManager()
        await mgr.release("unknown")  # Should not raise

    def test_active_count_unknown_tenant(self) -> None:
        """Unknown tenant has zero active sessions."""
        mgr = VoiceSessionManager()
        assert mgr.active_count("unknown") == 0


class TestCreateVoicePipeline:
    """Verify pipeline factory assembly."""

    def test_creates_all_components(self) -> None:
        """Factory returns rag_processor, stt_adapter, tts_adapter."""
        result = create_voice_pipeline(
            chat_service=MagicMock(),
            stt_provider=MagicMock(),
            tts_provider=MagicMock(),
            tenant_id="t1",
            user_id="u1",
            voice_config=TenantVoiceConfig(
                voice_enabled=True,
                stt_provider="whisper",
                tts_provider="piper",
                tts_voice="en_US-lessac-medium",
            ),
            tenant_model_config=MagicMock(),
        )

        assert "rag_processor" in result
        assert "stt_adapter" in result
        assert "tts_adapter" in result
        assert result["rag_processor"]._tenant_id == "t1"
