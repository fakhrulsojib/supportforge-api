"""Tests for voice-specific enums and exceptions.

Covers MessageChannel enum values and STTError/TTSError/VoiceBusyError
status codes and error codes.
"""

from __future__ import annotations

from app.core.exceptions import SupportForgeError
from app.domain.models.enums import MessageChannel


class TestMessageChannel:
    """Verify MessageChannel enum values."""

    def test_text_value(self) -> None:
        assert MessageChannel.TEXT == "text"
        assert MessageChannel.TEXT.value == "text"

    def test_voice_value(self) -> None:
        assert MessageChannel.VOICE == "voice"
        assert MessageChannel.VOICE.value == "voice"

    def test_is_str_enum(self) -> None:
        """MessageChannel is a str enum (for Pydantic/JSON compat)."""
        assert isinstance(MessageChannel.TEXT, str)

    def test_only_two_members(self) -> None:
        """Only TEXT and VOICE members exist."""
        assert len(MessageChannel) == 2


class TestSTTError:
    """Verify STTError exception."""

    def test_default_message(self) -> None:
        from app.core.exceptions import STTError
        err = STTError()
        assert err.message == "STT processing failed"
        assert err.status_code == 502
        assert err.error_code == "STT_ERROR"

    def test_custom_message(self) -> None:
        from app.core.exceptions import STTError
        err = STTError("Whisper crashed")
        assert err.message == "Whisper crashed"

    def test_is_supportforge_error(self) -> None:
        from app.core.exceptions import STTError
        assert issubclass(STTError, SupportForgeError)


class TestTTSError:
    """Verify TTSError exception."""

    def test_default_message(self) -> None:
        from app.core.exceptions import TTSError
        err = TTSError()
        assert err.message == "TTS synthesis failed"
        assert err.status_code == 502
        assert err.error_code == "TTS_ERROR"

    def test_custom_message(self) -> None:
        from app.core.exceptions import TTSError
        err = TTSError("Piper voice not found")
        assert err.message == "Piper voice not found"

    def test_is_supportforge_error(self) -> None:
        from app.core.exceptions import TTSError
        assert issubclass(TTSError, SupportForgeError)


class TestVoiceBusyError:
    """Verify VoiceBusyError exception."""

    def test_defaults(self) -> None:
        from app.core.exceptions import VoiceBusyError
        err = VoiceBusyError()
        assert err.status_code == 429
        assert err.error_code == "VOICE_BUSY"
        assert "busy" in err.message.lower()

    def test_is_supportforge_error(self) -> None:
        from app.core.exceptions import VoiceBusyError
        assert issubclass(VoiceBusyError, SupportForgeError)
