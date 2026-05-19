"""Tests for the SupportForge RAG Processor and Pipecat adapters.

All Pipecat classes are mocked since pipecat-ai is an optional dependency.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.infrastructure.voice.rag_processor import SupportForgeRAGProcessor


class TestSupportForgeRAGProcessor:
    """Verify the custom Pipecat FrameProcessor."""

    def _make_processor(self) -> SupportForgeRAGProcessor:
        """Create a processor with a mocked ChatService."""
        return SupportForgeRAGProcessor(
            chat_service=MagicMock(),
            tenant_id="tenant-1",
            user_id="user-1",
            conversation_id=None,
            tenant_model_config=MagicMock(),
            tenant_temperature=0.7,
            tenant_blocklist=None,
        )

    def test_initialization(self) -> None:
        """Processor initializes with all required fields."""
        proc = self._make_processor()
        assert proc._tenant_id == "tenant-1"
        assert proc._user_id == "user-1"
        assert proc._conversation_id is None

    @pytest.mark.asyncio
    async def test_process_transcript_calls_chat_service(self) -> None:
        """Transcript text is forwarded to ChatService.stream_message."""
        proc = self._make_processor()

        # Mock the streaming response
        async def _mock_stream(**kwargs):
            yield {"type": "token", "data": "Hello"}
            yield {"type": "done", "data": {"conversation_id": "conv-123"}}

        proc._chat_service.stream_message = _mock_stream

        # Collect output frames
        output_frames = []
        proc.push_frame = AsyncMock(side_effect=lambda f, *a: output_frames.append(f))

        await proc.process_transcript("How do I reset my password?")

        # Should have produced text frames
        assert len(output_frames) >= 1
        # conversation_id should be tracked
        assert proc._conversation_id == "conv-123"

    @pytest.mark.asyncio
    async def test_process_empty_transcript_noop(self) -> None:
        """Empty transcript is silently skipped."""
        proc = self._make_processor()
        proc.push_frame = AsyncMock()

        await proc.process_transcript("")
        proc.push_frame.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_whitespace_transcript_noop(self) -> None:
        """Whitespace-only transcript is silently skipped."""
        proc = self._make_processor()
        proc.push_frame = AsyncMock()

        await proc.process_transcript("   ")
        proc.push_frame.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_in_chat_service_produces_error_text(self) -> None:
        """ChatService errors produce a friendly error message."""
        proc = self._make_processor()

        async def _mock_stream(**kwargs):
            raise RuntimeError("RAG pipeline failed")
            yield  # pragma: no cover — make it a generator

        proc._chat_service.stream_message = _mock_stream
        output_frames = []
        proc.push_frame = AsyncMock(side_effect=lambda f, *a: output_frames.append(f))

        await proc.process_transcript("test query")

        # Should produce error text + end frame
        assert len(output_frames) >= 1
        # First frame should contain error text
        assert "error" in str(output_frames[0]).lower() or "sorry" in str(output_frames[0]).lower()

    @pytest.mark.asyncio
    async def test_conversation_id_persists_across_turns(self) -> None:
        """conversation_id from first turn is reused in subsequent turns."""
        proc = self._make_processor()

        async def _mock_stream(**kwargs):
            yield {"type": "done", "data": {"conversation_id": "conv-abc"}}

        proc._chat_service.stream_message = _mock_stream
        proc.push_frame = AsyncMock()

        await proc.process_transcript("first message")
        assert proc._conversation_id == "conv-abc"

        # Second message should use the existing conversation_id
        async def _mock_stream_2(**kwargs):
            assert kwargs.get("conversation_id") == "conv-abc"
            yield {"type": "done", "data": {"conversation_id": "conv-abc"}}

        proc._chat_service.stream_message = _mock_stream_2
        await proc.process_transcript("follow-up")

    @pytest.mark.asyncio
    async def test_source_frames_are_skipped(self) -> None:
        """Non-token, non-done frames don't produce output."""
        proc = self._make_processor()

        async def _mock_stream(**kwargs):
            yield {"type": "source", "data": [{"title": "doc.pdf"}]}
            yield {"type": "done", "data": {"conversation_id": "c1"}}

        proc._chat_service.stream_message = _mock_stream
        output_frames = []
        proc.push_frame = AsyncMock(side_effect=lambda f, *a: output_frames.append(f))

        await proc.process_transcript("test")

        # Only the end-of-response frame should appear, not source data
        token_frames = [f for f in output_frames if hasattr(f, "text")]
        assert len(token_frames) == 0


class TestPipecatSTTAdapter:
    """Verify the Pipecat STT adapter wrapper."""

    @pytest.mark.asyncio
    async def test_wraps_stt_provider(self) -> None:
        """Adapter forwards to STTProvider.transcribe."""
        from app.infrastructure.voice.pipecat_stt_adapter import PipecatSTTAdapter

        mock_provider = AsyncMock()
        mock_provider.transcribe = AsyncMock(return_value="hello world")

        adapter = PipecatSTTAdapter(mock_provider)
        result = await adapter.run_stt(b"\x00\x01")
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_stt_error_returns_empty(self) -> None:
        """STTError during transcription returns empty string."""
        from app.core.exceptions import STTError
        from app.infrastructure.voice.pipecat_stt_adapter import PipecatSTTAdapter

        mock_provider = AsyncMock()
        mock_provider.transcribe = AsyncMock(side_effect=STTError("failed"))

        adapter = PipecatSTTAdapter(mock_provider)
        result = await adapter.run_stt(b"\x00\x01")
        assert result == ""


class TestPipecatTTSAdapter:
    """Verify the Pipecat TTS adapter wrapper."""

    @pytest.mark.asyncio
    async def test_wraps_tts_provider(self) -> None:
        """Adapter forwards to TTSProvider.synthesize."""
        from app.infrastructure.voice.pipecat_tts_adapter import PipecatTTSAdapter

        mock_provider = AsyncMock()
        mock_provider.synthesize = AsyncMock(return_value=b"\x00\x01")

        adapter = PipecatTTSAdapter(mock_provider, voice="test-voice")
        result = await adapter.run_tts("hello")
        assert result == b"\x00\x01"
        mock_provider.synthesize.assert_called_once_with("hello", voice="test-voice")

    @pytest.mark.asyncio
    async def test_tts_error_returns_empty(self) -> None:
        """TTSError during synthesis returns empty bytes."""
        from app.core.exceptions import TTSError
        from app.infrastructure.voice.pipecat_tts_adapter import PipecatTTSAdapter

        mock_provider = AsyncMock()
        mock_provider.synthesize = AsyncMock(side_effect=TTSError("failed"))

        adapter = PipecatTTSAdapter(mock_provider)
        result = await adapter.run_tts("hello")
        assert result == b""
