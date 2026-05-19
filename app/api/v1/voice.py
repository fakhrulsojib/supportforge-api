"""Voice API endpoints — configuration, health, and session info.

Provides REST endpoints for:
- Checking voice availability for a tenant
- Voice service health status
- Active voice session counts (admin only)
- Toggling voice on/off per tenant (admin only)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from app.config import get_settings
from app.core.dependencies import get_current_user, require_role
from app.core.tenant_config import resolve_tenant_voice_config
from app.domain.models.enums import UserRole
from app.infrastructure.database.connection import get_async_session
from app.infrastructure.database.repositories.tenant_repo import SQLTenantRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.domain.models.user import User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/voice", tags=["Voice"])


@router.get("/config")
async def get_voice_config(
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> dict:
    """Return voice availability for the authenticated tenant."""
    logger.info(
        "voice_config_requested",
        tenant_id=user.tenant_id,
        user_id=user.id,
    )

    # Fetch tenant config_json from the database
    tenant_repo = SQLTenantRepository(session)
    tenant = await tenant_repo.get_by_id(user.tenant_id)
    tenant_config_json: dict = {}
    if tenant and tenant.config_json:
        tenant_config_json = tenant.config_json

    settings = get_settings()

    stt_available = hasattr(request.app.state, "stt_provider") and request.app.state.stt_provider is not None
    tts_available = hasattr(request.app.state, "tts_provider") and request.app.state.tts_provider is not None

    logger.debug(
        "voice_config_resolution_input",
        tenant_id=user.tenant_id,
        voice_enabled_in_config=tenant_config_json.get("voice_enabled"),
        stt_locally_available=stt_available,
        tts_locally_available=tts_available,
    )

    voice_config = resolve_tenant_voice_config(
        tenant_config_json,
        encryption_key=settings.secret_key,
        stt_locally_available=stt_available,
        tts_locally_available=tts_available,
    )

    result = {
        "voice_enabled": voice_config.voice_enabled,
        "stt_provider": voice_config.stt_provider,
        "tts_provider": voice_config.tts_provider,
        "tts_voice": voice_config.tts_voice,
        "max_voice_sessions": voice_config.max_voice_sessions,
    }

    logger.info(
        "voice_config_resolved",
        tenant_id=user.tenant_id,
        voice_enabled=voice_config.voice_enabled,
        stt_provider=voice_config.stt_provider,
        tts_provider=voice_config.tts_provider,
    )

    return result


@router.get("/health")
async def get_voice_health(
    request: Request,
    user: User = Depends(get_current_user),
) -> dict:
    """Return health status of STT and TTS services."""
    logger.info("voice_health_requested", tenant_id=user.tenant_id)

    stt_available = False
    tts_available = False

    stt_provider = getattr(request.app.state, "stt_provider", None)
    if stt_provider is not None:
        try:
            stt_available = await stt_provider.health_check()
            logger.debug("voice_stt_health", available=stt_available, provider=stt_provider.provider_name)
        except Exception:
            logger.warning("stt_health_check_failed", exc_info=True)

    tts_provider = getattr(request.app.state, "tts_provider", None)
    if tts_provider is not None:
        try:
            tts_available = await tts_provider.health_check()
            logger.debug("voice_tts_health", available=tts_available, provider=tts_provider.provider_name)
        except Exception:
            logger.warning("tts_health_check_failed", exc_info=True)

    result = {
        "stt_available": stt_available,
        "tts_available": tts_available,
        "stt_provider": getattr(stt_provider, "provider_name", None) if stt_provider else None,
        "tts_provider": getattr(tts_provider, "provider_name", None) if tts_provider else None,
    }

    logger.info("voice_health_result", **result)
    return result


@router.get("/sessions")
async def get_voice_sessions(
    request: Request,
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> dict:
    """Return active voice session count for the tenant (admin only)."""
    session_manager = getattr(request.app.state, "voice_session_manager", None)
    active = 0
    if session_manager is not None:
        active = session_manager.active_count(user.tenant_id)

    logger.info(
        "voice_sessions_queried",
        tenant_id=user.tenant_id,
        active_sessions=active,
    )

    return {
        "tenant_id": user.tenant_id,
        "active_sessions": active,
    }


@router.put("/toggle")
async def toggle_voice(
    request: Request,
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> dict:
    """Enable or disable voice for this tenant (admin only)."""
    from app.infrastructure.database.connection import AsyncSessionLocal
    from app.infrastructure.database.repositories.tenant_repo import SQLTenantRepository

    logger.info("voice_toggle_requested", tenant_id=user.tenant_id, user_id=user.id)

    async with AsyncSessionLocal() as session:
        repo = SQLTenantRepository(session)
        tenant = await repo.get_by_id(user.tenant_id)
        if not tenant:
            logger.warning("voice_toggle_tenant_not_found", tenant_id=user.tenant_id)
            return {"voice_enabled": False}

        config = dict(tenant.config_json or {})
        old_value = config.get("voice_enabled", False)
        new_value = not old_value
        config["voice_enabled"] = new_value
        await repo.update(user.tenant_id, config_json=config)
        await session.commit()

        logger.info(
            "voice_toggle_persisted",
            tenant_id=user.tenant_id,
            old_value=old_value,
            new_value=new_value,
        )

    # Re-resolve voice config to return current state
    stt_locally = getattr(request.app.state, "stt_provider", None) is not None
    tts_locally = getattr(request.app.state, "tts_provider", None) is not None

    voice_config = resolve_tenant_voice_config(
        config,
        stt_locally_available=stt_locally,
        tts_locally_available=tts_locally,
    )

    logger.info(
        "voice_toggle_result",
        tenant_id=user.tenant_id,
        voice_enabled=voice_config.voice_enabled,
        stt_provider=voice_config.stt_provider,
        tts_provider=voice_config.tts_provider,
    )

    return {
        "voice_enabled": voice_config.voice_enabled,
        "stt_provider": voice_config.stt_provider,
        "tts_provider": voice_config.tts_provider,
        "tts_voice": voice_config.tts_voice,
        "max_voice_sessions": voice_config.max_voice_sessions,
    }


@router.post("/transcribe")
async def transcribe_audio(
    request: Request,
    user: User = Depends(get_current_user),
) -> dict:
    """Transcribe an uploaded audio file to text via STT.

    Accepts multipart form upload with an ``audio`` field, or raw body.
    Audio is decoded via PyAV (webm, wav, mp3, ogg, etc) to PCM,
    then transcribed by faster-whisper.
    """
    stt_provider = getattr(request.app.state, "stt_provider", None)
    if stt_provider is None:
        logger.warning("transcribe_no_stt_provider", tenant_id=user.tenant_id)
        return {"error": "STT provider not available", "text": ""}

    # Read raw body (multipart or raw audio)
    content_type = request.headers.get("content-type", "")
    logger.info(
        "transcribe_request_received",
        tenant_id=user.tenant_id,
        user_id=user.id,
        content_type=content_type,
    )

    # Handle multipart form upload
    if "multipart" in content_type:
        form = await request.form()
        audio_file = form.get("audio")
        if not audio_file:
            logger.warning("transcribe_no_audio_field", tenant_id=user.tenant_id)
            return {"error": "No 'audio' field in form data", "text": ""}
        audio_bytes = await audio_file.read()
        filename = getattr(audio_file, "filename", "audio.webm")
    else:
        # Raw body upload
        audio_bytes = await request.body()
        filename = "audio.webm"

    if not audio_bytes:
        logger.warning("transcribe_empty_audio", tenant_id=user.tenant_id)
        return {"error": "Empty audio data", "text": ""}

    logger.info(
        "transcribe_audio_received",
        tenant_id=user.tenant_id,
        audio_bytes=len(audio_bytes),
        filename=filename,
    )

    # Decode webm/opus → PCM float32 numpy array using PyAV,
    # then pass the array directly to faster-whisper.
    # This avoids needing the ffmpeg CLI tool.
    try:
        import asyncio
        import io
        import time

        import av  # type: ignore[import-untyped]
        import numpy as np  # type: ignore[import-untyped]

        # Decode audio with PyAV
        logger.info("transcribe_decoding_audio", format="webm/opus")
        container = av.open(io.BytesIO(audio_bytes))
        audio_stream = next(s for s in container.streams if s.type == "audio")

        logger.info(
            "transcribe_audio_stream_info",
            codec=audio_stream.codec_context.name,
            sample_rate=audio_stream.codec_context.sample_rate,
            channels=audio_stream.codec_context.channels,
            duration_secs=round(float(audio_stream.duration * audio_stream.time_base), 2)
            if audio_stream.duration
            else "unknown",
        )

        # Resample to 16kHz mono float32 (what whisper expects)
        resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=16000,
        )

        pcm_chunks: list[bytes] = []
        for frame in container.decode(audio=0):
            resampled = resampler.resample(frame)
            for r in resampled:
                pcm_chunks.append(r.to_ndarray().tobytes())

        container.close()

        if not pcm_chunks:
            logger.warning("transcribe_no_audio_frames", tenant_id=user.tenant_id)
            return {"error": "No audio frames decoded", "text": ""}

        pcm_bytes = b"".join(pcm_chunks)
        # Convert Int16 PCM to float32 normalized [-1.0, 1.0]
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        logger.info(
            "transcribe_audio_decoded",
            pcm_bytes=len(pcm_bytes),
            samples=len(samples),
            duration_secs=round(len(samples) / 16000, 2),
        )

        # Run whisper transcription on the numpy array
        t0 = time.monotonic()
        segments, info = await asyncio.to_thread(
            stt_provider._model.transcribe,
            samples,
            language="en",
        )
        text = " ".join(s.text.strip() for s in segments)
        elapsed = round(time.monotonic() - t0, 3)

        logger.info(
            "transcribe_completed",
            tenant_id=user.tenant_id,
            text_length=len(text),
            text_preview=text[:100] if text else "(empty)",
            elapsed_secs=elapsed,
            language=getattr(info, "language", "en"),
            language_probability=round(getattr(info, "language_probability", 0), 3),
        )

        return {"text": text, "language": getattr(info, "language", "en")}

    except Exception as exc:
        logger.exception(
            "transcribe_failed",
            tenant_id=user.tenant_id,
            error=str(exc),
        )
        return {"error": f"Transcription failed: {exc}", "text": ""}


@router.post("/synthesize")
async def synthesize_text(
    request: Request,
    user: User = Depends(get_current_user),
) -> Response:
    """Synthesize text to speech using TTS provider.

    Accepts JSON ``{"text": "..."}`` and returns WAV audio.
    """

    tts_provider = getattr(request.app.state, "tts_provider", None)
    if tts_provider is None:
        logger.warning("synthesize_no_tts_provider", tenant_id=user.tenant_id)
        return Response(content=b"", media_type="audio/wav", status_code=503)

    body = await request.json()
    text = body.get("text", "").strip()

    if not text:
        logger.warning("synthesize_empty_text", tenant_id=user.tenant_id)
        return Response(content=b"", media_type="audio/wav")

    logger.info(
        "synthesize_request",
        tenant_id=user.tenant_id,
        text_length=len(text),
        text_preview=text[:80],
    )

    try:
        import struct
        import time

        t0 = time.monotonic()
        pcm_audio = await tts_provider.synthesize(text)
        elapsed = round(time.monotonic() - t0, 3)

        logger.info(
            "synthesize_completed",
            tenant_id=user.tenant_id,
            pcm_bytes=len(pcm_audio),
            elapsed_secs=elapsed,
        )

        # Wrap raw PCM in a WAV header for browser playback
        sample_rate = 22050
        channels = 1
        bits_per_sample = 16
        data_size = len(pcm_audio)
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,  # PCM
            channels,
            sample_rate,
            sample_rate * channels * bits_per_sample // 8,
            channels * bits_per_sample // 8,
            bits_per_sample,
            b"data",
            data_size,
        )

        wav_audio = header + pcm_audio

        logger.info(
            "synthesize_wav_ready",
            tenant_id=user.tenant_id,
            wav_bytes=len(wav_audio),
            duration_secs=round(data_size / (sample_rate * channels * bits_per_sample // 8), 2),
        )

        return Response(content=wav_audio, media_type="audio/wav")

    except Exception as exc:
        logger.exception(
            "synthesize_failed",
            tenant_id=user.tenant_id,
            error=str(exc),
        )
        return Response(content=b"", media_type="audio/wav", status_code=500)
