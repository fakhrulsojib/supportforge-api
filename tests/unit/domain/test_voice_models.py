"""Tests for voice domain models.

Verifies VoiceFrame (frozen dataclass), VoiceFrameType constants,
and MessageChannel enum.
"""

from __future__ import annotations

import pytest

from app.domain.models.voice import VoiceFrame, VoiceFrameType


class TestVoiceFrameType:
    """Verify VoiceFrameType constants are correct."""

    def test_all_frame_types_exist(self) -> None:
        """All expected frame type constants exist."""
        assert VoiceFrameType.TRANSCRIPT == "transcript"
        assert VoiceFrameType.TEXT_TOKEN == "text_token"
        assert VoiceFrameType.AUDIO == "audio"
        assert VoiceFrameType.SOURCE == "source"
        assert VoiceFrameType.DONE == "done"
        assert VoiceFrameType.ERROR == "error"
        assert VoiceFrameType.CONFIG == "config"
        assert VoiceFrameType.VOICE_UNAVAILABLE == "voice_unavailable"
        assert VoiceFrameType.VOICE_BUSY == "voice_busy"


class TestVoiceFrame:
    """Verify VoiceFrame is a frozen, slotted dataclass."""

    def test_creation(self) -> None:
        """VoiceFrame can be created with required fields."""
        frame = VoiceFrame(type="transcript", data="hello")
        assert frame.type == "transcript"
        assert frame.data == "hello"
        assert frame.is_binary is False

    def test_binary_frame(self) -> None:
        """VoiceFrame can represent binary data."""
        frame = VoiceFrame(type="audio", data=b"\x00\x01", is_binary=True)
        assert frame.is_binary is True
        assert isinstance(frame.data, bytes)

    def test_frozen(self) -> None:
        """VoiceFrame is immutable (frozen)."""
        frame = VoiceFrame(type="done", data=None)
        with pytest.raises(AttributeError):
            frame.type = "error"  # type: ignore[misc]

    def test_slots(self) -> None:
        """VoiceFrame uses __slots__ (no __dict__)."""
        frame = VoiceFrame(type="done", data=None)
        assert not hasattr(frame, "__dict__")

    def test_equality(self) -> None:
        """VoiceFrame instances with same values are equal."""
        a = VoiceFrame(type="error", data="oops")
        b = VoiceFrame(type="error", data="oops")
        assert a == b

    def test_inequality(self) -> None:
        """VoiceFrame instances with different values are not equal."""
        a = VoiceFrame(type="error", data="oops")
        b = VoiceFrame(type="done", data="oops")
        assert a != b
