"""Tests for TTS infrastructure — factory and PiperAdapter.

Uses mocks for piper-tts to avoid requiring the actual library.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import TTSError
from app.domain.interfaces.tts_provider import TTSProvider


class TestTTSFactory:
    """Verify the TTS provider factory."""

    def test_piper_returns_tts_provider(self) -> None:
        """Factory creates a PiperAdapter for 'piper' provider."""
        from app.infrastructure.tts.factory import get_tts_provider

        with patch("app.infrastructure.tts.piper_adapter.PiperAdapter._load_voice"):
            provider = get_tts_provider("piper", voice="en_US-lessac-medium")
        assert isinstance(provider, TTSProvider)
        assert provider.provider_name == "piper"

    def test_unknown_provider_raises(self) -> None:
        """Factory raises ValueError for unknown provider names."""
        from app.infrastructure.tts.factory import get_tts_provider

        with pytest.raises(ValueError, match="Unknown TTS provider"):
            get_tts_provider("nonexistent")

    def test_elevenlabs_not_installed(self) -> None:
        """Factory raises ImportError when elevenlabs SDK is not installed."""
        from app.infrastructure.tts.factory import get_tts_provider

        with pytest.raises((ValueError, ImportError)):
            get_tts_provider("elevenlabs", api_key="test")


class TestPiperAdapter:
    """Verify the PiperAdapter implementation."""

    def _make_adapter(self) -> object:
        """Create a PiperAdapter with a mocked voice model."""
        from app.infrastructure.tts.piper_adapter import PiperAdapter

        with patch.object(PiperAdapter, "_load_voice"):
            adapter = PiperAdapter(voice="en_US-lessac-medium")
        adapter._voice_model = MagicMock()
        return adapter

    def test_provider_name(self) -> None:
        """provider_name returns 'piper'."""
        adapter = self._make_adapter()
        assert adapter.provider_name == "piper"

    @pytest.mark.asyncio
    async def test_synthesize_empty_text_returns_empty(self) -> None:
        """Empty text returns empty bytes (no crash)."""
        adapter = self._make_adapter()
        result = await adapter.synthesize("")
        assert result == b""

    @pytest.mark.asyncio
    async def test_synthesize_whitespace_only_returns_empty(self) -> None:
        """Whitespace-only text returns empty bytes."""
        adapter = self._make_adapter()
        result = await adapter.synthesize("   ")
        assert result == b""

    @pytest.mark.asyncio
    async def test_synthesize_success(self) -> None:
        """Valid text returns non-empty audio bytes."""
        adapter = self._make_adapter()

        # Mock piper synthesize to return raw audio
        adapter._voice_model.synthesize = MagicMock(
            return_value=iter([b"\x00\x01\x02\x03"])
        )

        async def _fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("app.infrastructure.tts.piper_adapter.asyncio.to_thread", side_effect=_fake_to_thread):
            result = await adapter.synthesize("hello world")

        assert isinstance(result, bytes)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_synthesize_model_error_wraps_in_tts_error(self) -> None:
        """Model errors are wrapped in TTSError."""
        adapter = self._make_adapter()
        adapter._voice_model.synthesize = MagicMock(side_effect=RuntimeError("voice failed"))

        async def _fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with (
            patch("app.infrastructure.tts.piper_adapter.asyncio.to_thread", side_effect=_fake_to_thread),
            pytest.raises(TTSError, match="Synthesis failed"),
        ):
            await adapter.synthesize("test")

    @pytest.mark.asyncio
    async def test_warm_up_calls_load_voice(self) -> None:
        """warm_up() triggers voice model loading."""
        from app.infrastructure.tts.piper_adapter import PiperAdapter

        with patch.object(PiperAdapter, "_load_voice") as mock_load:
            adapter = PiperAdapter(voice="en_US-lessac-medium")
            await adapter.warm_up()
            assert mock_load.call_count >= 1

    @pytest.mark.asyncio
    async def test_health_check_with_model(self) -> None:
        """health_check returns True when voice model is loaded."""
        adapter = self._make_adapter()
        result = await adapter.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_without_model(self) -> None:
        """health_check returns False when voice model is None."""
        adapter = self._make_adapter()
        adapter._voice_model = None
        result = await adapter.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_list_voices_returns_list(self) -> None:
        """list_voices returns a non-empty list when model is loaded."""
        adapter = self._make_adapter()
        voices = await adapter.list_voices()
        assert isinstance(voices, list)
        assert len(voices) >= 1
        assert "id" in voices[0]
        assert "name" in voices[0]
