"""Tests for the STT provider domain interface.

Verifies the ABC contract: abstract methods, warm-up, health check.
"""

from __future__ import annotations

import pytest

from app.domain.interfaces.stt_provider import STTProvider


class _ConcreteSTT(STTProvider):
    """Minimal concrete implementation for ABC enforcement tests."""

    @property
    def provider_name(self) -> str:
        return "test"

    async def transcribe(
        self, audio: bytes, *, sample_rate: int = 16000, language: str = "en",
    ) -> str:
        return "hello world"

    async def warm_up(self) -> None:
        pass

    async def health_check(self) -> bool:
        return True


class TestSTTProviderABC:
    """Verify STTProvider is a proper ABC with enforced methods."""

    def test_cannot_instantiate_abstract(self) -> None:
        """STTProvider cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            STTProvider()  # type: ignore[abstract]

    def test_concrete_instantiation(self) -> None:
        """A fully-implemented subclass can be instantiated."""
        stt = _ConcreteSTT()
        assert stt.provider_name == "test"

    @pytest.mark.asyncio
    async def test_transcribe_returns_string(self) -> None:
        """transcribe() returns a string transcript."""
        stt = _ConcreteSTT()
        result = await stt.transcribe(b"\x00\x01\x02")
        assert isinstance(result, str)
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_warm_up_callable(self) -> None:
        """warm_up() can be called without error."""
        stt = _ConcreteSTT()
        await stt.warm_up()  # should not raise

    @pytest.mark.asyncio
    async def test_health_check_returns_bool(self) -> None:
        """health_check() returns a boolean."""
        stt = _ConcreteSTT()
        result = await stt.health_check()
        assert result is True
