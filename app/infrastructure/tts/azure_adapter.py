"""Azure Cognitive Services TTS adapter.

Uses the ``azure-cognitiveservices-speech`` SDK for cloud-based
text-to-speech.  Runs inference in a thread pool to avoid blocking
the async event loop (the SDK is synchronous).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

from app.core.exceptions import TTSError
from app.domain.interfaces.tts_provider import TTSProvider

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = structlog.get_logger(__name__)

# Common Azure neural voices — natural-sounding, free-tier eligible
AZURE_VOICES = [
    {"id": "en-US-AriaNeural", "name": "Aria (US English, Female)"},
    {"id": "en-US-GuyNeural", "name": "Guy (US English, Male)"},
    {"id": "en-US-JennyNeural", "name": "Jenny (US English, Female)"},
    {"id": "en-GB-SoniaNeural", "name": "Sonia (UK English, Female)"},
    {"id": "en-GB-RyanNeural", "name": "Ryan (UK English, Male)"},
    {"id": "en-AU-NatashaNeural", "name": "Natasha (Australian English, Female)"},
]


class AzureTTSAdapter(TTSProvider):
    """TTS adapter using Azure Cognitive Services Speech.

    Args:
        subscription_key: Azure Speech resource subscription key.
        region: Azure region (e.g. ``eastus``, ``westus2``).
        voice: Azure voice name (default ``en-US-AriaNeural``).
    """

    def __init__(
        self,
        subscription_key: str,
        region: str = "eastus",
        voice: str = "en-US-AriaNeural",
    ) -> None:
        self._subscription_key = subscription_key
        self._region = region
        self._default_voice = voice
        self._sdk: Any = None
        logger.info(
            "azure_tts_adapter_init",
            region=region,
            voice=voice,
            key_length=len(subscription_key) if subscription_key else 0,
            key_preview=f"{subscription_key[:4]}...{subscription_key[-4:]}"
            if subscription_key and len(subscription_key) > 8
            else "(short key)",
        )

    @property
    def provider_name(self) -> str:
        """Return ``'azure'``."""
        return "azure"

    def _ensure_sdk(self) -> Any:
        """Lazy-import the Azure Speech SDK."""
        if self._sdk is None:
            logger.debug("azure_tts_sdk_loading")
            try:
                import azure.cognitiveservices.speech as speechsdk  # type: ignore[import-untyped]

                self._sdk = speechsdk
                logger.info(
                    "azure_tts_sdk_loaded",
                    sdk_version=getattr(speechsdk, "__version__", "unknown"),
                )
            except ImportError:
                msg = (
                    "azure-cognitiveservices-speech is not installed. "
                    "Install with: pip install azure-cognitiveservices-speech"
                )
                logger.error("azure_tts_sdk_import_failed", error=msg)
                raise TTSError(msg) from None
        return self._sdk

    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        *,
        sample_rate: int = 22050,
    ) -> bytes:
        """Synthesize text to raw PCM Int16 mono audio via Azure.

        Args:
            text: Text to synthesize.
            voice: Azure voice name. ``"default"`` uses the configured voice.
            sample_rate: Ignored — Azure returns at its native rate (16kHz).
                The WAV wrapper in the endpoint handles this.

        Returns:
            Raw PCM Int16 mono audio bytes.

        Raises:
            TTSError: On synthesis failure.
        """
        if not text:
            logger.debug("azure_tts_synthesize_skip_empty")
            return b""

        speechsdk = self._ensure_sdk()
        voice_name = self._default_voice if voice == "default" else voice

        logger.info(
            "azure_tts_synthesize_start",
            text_length=len(text),
            text_preview=text[:80],
            voice=voice_name,
            region=self._region,
            output_format="Raw16Khz16BitMonoPcm",
        )

        def _sync_synthesize() -> bytes:
            """Run Azure TTS synchronously (SDK is blocking)."""
            import time

            # ── Step 1: Build SpeechConfig ────────────────────────
            t_config = time.monotonic()
            speech_config = speechsdk.SpeechConfig(
                subscription=self._subscription_key,
                region=self._region,
            )
            speech_config.speech_synthesis_voice_name = voice_name
            # Request raw PCM Int16 mono — no WAV header from Azure
            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm,
            )
            logger.debug(
                "azure_tts_config_built",
                voice=voice_name,
                output_format="Raw16Khz16BitMonoPcm",
                config_ms=round((time.monotonic() - t_config) * 1000, 1),
            )

            # ── Step 2: Create synthesizer and run ────────────────
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=None,  # no speaker output
            )
            logger.debug("azure_tts_synthesizer_created")

            logger.debug(
                "azure_tts_speak_async_start",
                text_length=len(text),
            )
            t_speak = time.monotonic()
            result = synthesizer.speak_text_async(text).get()
            speak_ms = round((time.monotonic() - t_speak) * 1000, 1)

            # ── Step 3: Handle result ─────────────────────────────
            logger.info(
                "azure_tts_speak_result",
                reason=str(result.reason),
                speak_ms=speak_ms,
                audio_data_bytes=len(result.audio_data) if result.audio_data else 0,
            )

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                audio_data = result.audio_data
                # Estimate duration: 16kHz * 2 bytes/sample = 32000 bytes/sec
                est_duration_secs = round(len(audio_data) / 32000, 2) if audio_data else 0
                logger.info(
                    "azure_tts_synthesized",
                    audio_bytes=len(audio_data),
                    estimated_duration_secs=est_duration_secs,
                    voice=voice_name,
                    speak_ms=speak_ms,
                )
                return audio_data
            if result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                logger.error(
                    "azure_tts_canceled",
                    cancellation_reason=str(cancellation.reason),
                    error_code=str(getattr(cancellation, "error_code", "N/A")),
                    error_details=cancellation.error_details,
                    voice=voice_name,
                    speak_ms=speak_ms,
                )
                raise TTSError(f"Azure TTS error: {cancellation.error_details}")

            logger.error(
                "azure_tts_unexpected_reason",
                reason=str(result.reason),
                speak_ms=speak_ms,
            )
            raise TTSError(f"Azure TTS unexpected result: {result.reason}")

        try:
            import time

            t0 = time.monotonic()
            pcm_audio = await asyncio.to_thread(_sync_synthesize)
            elapsed = round(time.monotonic() - t0, 3)

            # Final summary log
            est_duration_secs = round(len(pcm_audio) / 32000, 2) if pcm_audio else 0
            logger.info(
                "azure_tts_synthesize_done",
                audio_bytes=len(pcm_audio),
                estimated_duration_secs=est_duration_secs,
                total_elapsed_secs=elapsed,
                voice=voice_name,
                realtime_factor=round(elapsed / est_duration_secs, 2) if est_duration_secs > 0 else 0,
                text_length=len(text),
            )

            return pcm_audio
        except TTSError:
            raise
        except Exception as exc:
            msg = f"Azure TTS failed: {exc}"
            logger.exception(
                "azure_tts_error",
                error_type=type(exc).__name__,
                error_message=str(exc),
                text_length=len(text),
                voice=voice_name,
                region=self._region,
            )
            raise TTSError(msg) from exc

    async def synthesize_stream(
        self,
        text: str,
        voice: str = "default",
        *,
        sample_rate: int = 22050,
    ) -> AsyncGenerator[bytes, None]:
        """Stream synthesis — falls back to single-shot for Azure SDK.

        The Azure SDK does support streaming, but for simplicity we
        synthesize the full audio and yield it as a single chunk.

        Yields:
            PCM Int16 mono audio byte chunks.
        """
        logger.debug(
            "azure_tts_stream_start",
            text_length=len(text),
            voice=voice,
        )
        audio = await self.synthesize(text, voice, sample_rate=sample_rate)
        if audio:
            logger.debug(
                "azure_tts_stream_yielding",
                chunk_bytes=len(audio),
            )
            yield audio
        logger.debug("azure_tts_stream_end")

    async def list_voices(self) -> list[dict[str, str]]:
        """Return common Azure neural voices.

        For a full list, Azure provides a REST API, but we return
        the most popular English voices for simplicity.
        """
        logger.debug("azure_tts_list_voices", count=len(AZURE_VOICES))
        return AZURE_VOICES

    async def warm_up(self) -> None:
        """No-op — cloud service, no model to pre-load."""
        self._ensure_sdk()
        logger.info(
            "azure_tts_warm_up",
            region=self._region,
            voice=self._default_voice,
            status="ready",
        )

    async def health_check(self) -> bool:
        """Check if the Azure SDK is available."""
        try:
            self._ensure_sdk()
            logger.debug("azure_tts_health_check", status="healthy")
            return True
        except Exception as exc:
            logger.warning(
                "azure_tts_health_check_failed",
                error=str(exc),
            )
            return False
