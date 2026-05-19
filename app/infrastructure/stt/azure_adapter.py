"""Azure Cognitive Services STT adapter.

Uses the ``azure-cognitiveservices-speech`` SDK for cloud-based
speech-to-text.  Runs inference in a thread pool to avoid blocking
the async event loop (the SDK is synchronous).
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from app.core.exceptions import STTError
from app.domain.interfaces.stt_provider import STTProvider

logger = structlog.get_logger(__name__)


class AzureSTTAdapter(STTProvider):
    """STT adapter using Azure Cognitive Services Speech.

    Args:
        subscription_key: Azure Speech resource subscription key.
        region: Azure region (e.g. ``eastus``, ``westus2``).
    """

    def __init__(
        self,
        subscription_key: str,
        region: str = "eastus",
    ) -> None:
        self._subscription_key = subscription_key
        self._region = region
        self._sdk: Any = None
        logger.info(
            "azure_stt_adapter_init",
            region=region,
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
            logger.debug("azure_stt_sdk_loading")
            try:
                import azure.cognitiveservices.speech as speechsdk  # type: ignore[import-untyped]

                self._sdk = speechsdk
                logger.info(
                    "azure_stt_sdk_loaded",
                    sdk_version=getattr(speechsdk, "__version__", "unknown"),
                )
            except ImportError:
                msg = (
                    "azure-cognitiveservices-speech is not installed. "
                    "Install with: pip install azure-cognitiveservices-speech"
                )
                logger.error("azure_stt_sdk_import_failed", error=msg)
                raise STTError(msg) from None
        return self._sdk

    async def transcribe(
        self,
        audio: bytes,
        *,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> str:
        """Transcribe raw PCM Int16 mono audio to text via Azure.

        Args:
            audio: Raw PCM Int16 mono audio bytes.
            sample_rate: Audio sample rate in Hz (default 16000).
            language: BCP-47 language code (default ``"en"``).

        Returns:
            Transcribed text string.

        Raises:
            STTError: On transcription failure.
        """
        if not audio:
            logger.debug("azure_stt_transcribe_skip_empty")
            return ""

        speechsdk = self._ensure_sdk()

        # Calculate audio duration for logging
        audio_duration_secs = round(len(audio) / (sample_rate * 2), 2)  # 2 bytes per sample (Int16)

        logger.info(
            "azure_stt_transcribe_start",
            audio_bytes=len(audio),
            audio_duration_secs=audio_duration_secs,
            sample_rate=sample_rate,
            language=language,
            region=self._region,
        )

        def _sync_transcribe() -> str:
            """Run Azure STT synchronously (SDK is blocking)."""
            import time

            # ── Step 1: Build SpeechConfig ────────────────────────
            t_config = time.monotonic()
            speech_config = speechsdk.SpeechConfig(
                subscription=self._subscription_key,
                region=self._region,
            )
            # Map simple language codes to BCP-47 if needed
            lang_map = {"en": "en-US", "es": "es-ES", "fr": "fr-FR", "de": "de-DE"}
            resolved_lang = lang_map.get(language, language)
            speech_config.speech_recognition_language = resolved_lang
            logger.debug(
                "azure_stt_config_built",
                resolved_language=resolved_lang,
                config_ms=round((time.monotonic() - t_config) * 1000, 1),
            )

            # ── Step 2: Create PushAudioInputStream ───────────────
            t_stream = time.monotonic()
            audio_format = speechsdk.audio.AudioStreamFormat(
                samples_per_second=sample_rate,
                bits_per_sample=16,
                channels=1,
            )
            push_stream = speechsdk.audio.PushAudioInputStream(
                stream_format=audio_format,
            )
            push_stream.write(audio)
            push_stream.close()

            audio_config = speechsdk.audio.AudioConfig(
                stream=push_stream,
            )
            logger.debug(
                "azure_stt_audio_stream_ready",
                stream_bytes_written=len(audio),
                stream_ms=round((time.monotonic() - t_stream) * 1000, 1),
            )

            # ── Step 3: Create recognizer and run ─────────────────
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config,
            )

            logger.debug("azure_stt_recognize_once_start")
            t_recognize = time.monotonic()
            result = recognizer.recognize_once()
            recognize_ms = round((time.monotonic() - t_recognize) * 1000, 1)

            # ── Step 4: Handle result ─────────────────────────────
            logger.info(
                "azure_stt_recognize_once_result",
                reason=str(result.reason),
                recognize_ms=recognize_ms,
                result_text_length=len(result.text) if hasattr(result, "text") else 0,
            )

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                logger.info(
                    "azure_stt_recognized",
                    text=result.text,
                    text_length=len(result.text),
                    recognize_ms=recognize_ms,
                )
                return result.text
            if result.reason == speechsdk.ResultReason.NoMatch:
                no_match_detail = result.no_match_details if hasattr(result, "no_match_details") else None
                logger.warning(
                    "azure_stt_no_match",
                    reason=str(no_match_detail.reason) if no_match_detail else "unknown",
                    audio_duration_secs=audio_duration_secs,
                    recognize_ms=recognize_ms,
                )
                return ""
            if result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                logger.error(
                    "azure_stt_canceled",
                    cancellation_reason=str(cancellation.reason),
                    error_code=str(getattr(cancellation, "error_code", "N/A")),
                    error_details=cancellation.error_details,
                    recognize_ms=recognize_ms,
                )
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    raise STTError(f"Azure STT error: {cancellation.error_details}")
                return ""

            logger.warning(
                "azure_stt_unexpected_reason",
                reason=str(result.reason),
                recognize_ms=recognize_ms,
            )
            return ""

        try:
            import time

            t0 = time.monotonic()
            text = await asyncio.to_thread(_sync_transcribe)
            elapsed = round(time.monotonic() - t0, 3)

            logger.info(
                "azure_stt_transcribe_done",
                text_length=len(text),
                text_preview=text[:100] if text else "(empty)",
                total_elapsed_secs=elapsed,
                audio_duration_secs=audio_duration_secs,
                realtime_factor=round(elapsed / audio_duration_secs, 2) if audio_duration_secs > 0 else 0,
            )

            return text
        except STTError:
            raise
        except Exception as exc:
            msg = f"Azure STT failed: {exc}"
            logger.exception(
                "azure_stt_error",
                error_type=type(exc).__name__,
                error_message=str(exc),
                audio_bytes=len(audio),
                region=self._region,
            )
            raise STTError(msg) from exc

    async def warm_up(self) -> None:
        """No-op — cloud service, no model to pre-load."""
        self._ensure_sdk()
        logger.info(
            "azure_stt_warm_up",
            region=self._region,
            status="ready",
        )

    async def health_check(self) -> bool:
        """Check if the Azure SDK is available."""
        try:
            self._ensure_sdk()
            logger.debug("azure_stt_health_check", status="healthy")
            return True
        except Exception as exc:
            logger.warning(
                "azure_stt_health_check_failed",
                error=str(exc),
            )
            return False
