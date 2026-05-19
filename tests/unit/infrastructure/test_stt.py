"""Tests for STT infrastructure — factory and WhisperAdapter.

Uses mocks for faster-whisper to avoid requiring the actual library.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import STTError
from app.domain.interfaces.stt_provider import STTProvider


class TestSTTFactory:
    """Verify the STT provider factory."""

    def test_whisper_returns_stt_provider(self) -> None:
        """Factory creates a WhisperAdapter for 'whisper' provider."""
        from app.infrastructure.stt.factory import get_stt_provider

        with patch("app.infrastructure.stt.whisper_adapter.WhisperAdapter._load_model"):
            provider = get_stt_provider("whisper", model_size="tiny")
        assert isinstance(provider, STTProvider)
        assert provider.provider_name == "whisper"

    def test_unknown_provider_raises(self) -> None:
        """Factory raises ValueError for unknown provider names."""
        from app.infrastructure.stt.factory import get_stt_provider

        with pytest.raises(ValueError, match="Unknown STT provider"):
            get_stt_provider("nonexistent")

    def test_deepgram_not_installed(self) -> None:
        """Factory raises ValueError when deepgram adapter is not available."""
        from app.infrastructure.stt.factory import get_stt_provider

        with pytest.raises(ValueError, match="not available"):
            get_stt_provider("deepgram", api_key="test")


class TestWhisperAdapter:
    """Verify the WhisperAdapter implementation."""

    def _make_adapter(self) -> object:
        """Create a WhisperAdapter with a mocked model."""
        from app.infrastructure.stt.whisper_adapter import WhisperAdapter

        with patch.object(WhisperAdapter, "_load_model"):
            adapter = WhisperAdapter(model_size="tiny")
        adapter._model = MagicMock()
        return adapter

    def test_provider_name(self) -> None:
        """provider_name returns 'whisper'."""
        adapter = self._make_adapter()
        assert adapter.provider_name == "whisper"

    @pytest.mark.asyncio
    async def test_transcribe_empty_audio_raises(self) -> None:
        """Empty audio buffer raises STTError."""
        adapter = self._make_adapter()
        with pytest.raises(STTError, match="Empty audio"):
            await adapter.transcribe(b"")

    @pytest.mark.asyncio
    async def test_transcribe_oversized_audio_raises(self) -> None:
        """Audio exceeding MAX_AUDIO_BYTES raises STTError."""
        adapter = self._make_adapter()
        # 11MB of zeros
        huge_audio = b"\x00" * (11 * 1024 * 1024)
        with pytest.raises(STTError, match="exceeds"):
            await adapter.transcribe(huge_audio)

    @pytest.mark.asyncio
    async def test_transcribe_success(self) -> None:
        """Valid audio returns transcript string."""
        adapter = self._make_adapter()

        # Mock faster-whisper segment
        mock_segment = MagicMock()
        mock_segment.text = " hello world "
        adapter._model.transcribe = MagicMock(return_value=([mock_segment], None))

        async def _fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("app.infrastructure.stt.whisper_adapter.asyncio.to_thread", side_effect=_fake_to_thread):
            result = await adapter.transcribe(b"\x00\x01" * 100)

        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_transcribe_model_error_wraps_in_stt_error(self) -> None:
        """Model errors are wrapped in STTError."""
        adapter = self._make_adapter()
        adapter._model.transcribe = MagicMock(side_effect=RuntimeError("GPU OOM"))

        async def _fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("app.infrastructure.stt.whisper_adapter.asyncio.to_thread", side_effect=_fake_to_thread):
            with pytest.raises(STTError, match="Transcription failed"):
                await adapter.transcribe(b"\x00\x01" * 100)

    @pytest.mark.asyncio
    async def test_warm_up_calls_load_model(self) -> None:
        """warm_up() triggers model loading."""
        from app.infrastructure.stt.whisper_adapter import WhisperAdapter

        with patch.object(WhisperAdapter, "_load_model") as mock_load:
            adapter = WhisperAdapter(model_size="tiny")
            await adapter.warm_up()
            assert mock_load.call_count >= 1

    @pytest.mark.asyncio
    async def test_health_check_with_model(self) -> None:
        """health_check returns True when model is loaded."""
        adapter = self._make_adapter()
        result = await adapter.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_without_model(self) -> None:
        """health_check returns False when model is None."""
        adapter = self._make_adapter()
        adapter._model = None
        result = await adapter.health_check()
        assert result is False

