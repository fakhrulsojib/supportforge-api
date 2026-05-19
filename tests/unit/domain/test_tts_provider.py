"""Tests for the TTS provider domain interface.

Verifies the ABC contract: synthesize, synthesize_stream, list_voices,
warm_up, health_check.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from app.domain.interfaces.tts_provider import TTSProvider


class _ConcreteTTS(TTSProvider):
    """Minimal concrete implementation for ABC enforcement tests."""

    @property
    def provider_name(self) -> str:
        return "test-tts"

    async def synthesize(
        self, text: str, voice: str = "default", *, sample_rate: int = 22050,
    ) -> bytes:
        return b"\x00\x01"

    async def synthesize_stream(
        self, text: str, voice: str = "default", *, sample_rate: int = 22050,
    ) -> AsyncGenerator[bytes, None]:
        yield b"\x00"
        yield b"\x01"

    async def list_voices(self) -> list[dict[str, str]]:
        return [{"id": "default", "name": "Default Voice"}]

    async def warm_up(self) -> None:
        pass

    async def health_check(self) -> bool:
        return True


class TestTTSProviderABC:
    """Verify TTSProvider is a proper ABC with enforced methods."""

    def test_cannot_instantiate_abstract(self) -> None:
        """TTSProvider cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            TTSProvider()  # type: ignore[abstract]

    def test_concrete_instantiation(self) -> None:
        """A fully-implemented subclass can be instantiated."""
        tts = _ConcreteTTS()
        assert tts.provider_name == "test-tts"

    @pytest.mark.asyncio
    async def test_synthesize_returns_bytes(self) -> None:
        """synthesize() returns audio bytes."""
        tts = _ConcreteTTS()
        result = await tts.synthesize("hello")
        assert isinstance(result, bytes)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_synthesize_stream_yields_bytes(self) -> None:
        """synthesize_stream() yields byte chunks."""
        tts = _ConcreteTTS()
        chunks: list[bytes] = []
        async for chunk in tts.synthesize_stream("hello"):
            chunks.append(chunk)
        assert len(chunks) == 2
        assert all(isinstance(c, bytes) for c in chunks)

    @pytest.mark.asyncio
    async def test_list_voices_returns_list(self) -> None:
        """list_voices() returns a list of dicts with id and name."""
        tts = _ConcreteTTS()
        voices = await tts.list_voices()
        assert len(voices) == 1
        assert voices[0]["id"] == "default"
        assert "name" in voices[0]

    @pytest.mark.asyncio
    async def test_warm_up_callable(self) -> None:
        """warm_up() can be called without error."""
        tts = _ConcreteTTS()
        await tts.warm_up()  # should not raise

    @pytest.mark.asyncio
    async def test_health_check_returns_bool(self) -> None:
        """health_check() returns a boolean."""
        tts = _ConcreteTTS()
        result = await tts.health_check()
        assert result is True
