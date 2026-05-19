"""Voice domain models — value objects for the voice pipeline.

Pure domain — ZERO framework imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class VoiceFrameType:
    """Constants for voice pipeline frame types.

    Used by the Pipecat pipeline and WebRTC transport to
    identify frame payloads without magic strings.
    """

    TRANSCRIPT = "transcript"
    TEXT_TOKEN = "text_token"
    AUDIO = "audio"
    SOURCE = "source"
    DONE = "done"
    ERROR = "error"
    CONFIG = "config"
    VOICE_UNAVAILABLE = "voice_unavailable"
    VOICE_BUSY = "voice_busy"


@dataclass(frozen=True, slots=True)
class VoiceFrame:
    """Immutable value object yielded by Pipecat pipeline.

    Attributes:
        type: Frame type constant from VoiceFrameType.
        data: Payload — string for text, bytes for audio, dict for metadata.
        is_binary: True if data is raw audio bytes.
    """

    type: str
    data: Any
    is_binary: bool = False
