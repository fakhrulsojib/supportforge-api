"""Voice API endpoints — configuration, health, and session info.

Provides REST endpoints for:
- Checking voice availability for a tenant
- Voice service health status
- Active voice session counts (admin only)
- Toggling voice on/off per tenant (admin only)
- Saving per-tenant voice provider config (admin only)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


def _resolve_stt_for_tenant(
    voice_config: Any,
    app_state: Any,
) -> Any:
    """Resolve the STT provider for a tenant.

    If the tenant has a cloud STT provider configured with an API key,
    create a cloud adapter instance.  Otherwise fall back to the
    global local provider (Whisper) from app.state.
    """
    logger.debug(
        "stt_resolve_start",
        configured_provider=voice_config.stt_provider,
        has_api_key=bool(voice_config.stt_api_key),
        azure_region=getattr(voice_config, "azure_region", None),
    )

    if (
        voice_config.stt_provider
        and voice_config.stt_provider != "whisper"
        and voice_config.stt_api_key
    ):
        from app.infrastructure.stt.factory import get_stt_provider

        logger.info(
            "stt_resolve_cloud",
            provider=voice_config.stt_provider,
            region=getattr(voice_config, "azure_region", "eastus"),
        )
        try:
            provider = get_stt_provider(
                voice_config.stt_provider,
                subscription_key=voice_config.stt_api_key,
                region=getattr(voice_config, "azure_region", "eastus"),
            )
            logger.info(
                "stt_resolve_cloud_ok",
                provider=provider.provider_name,
            )
            return provider
        except Exception:
            logger.warning(
                "tenant_cloud_stt_init_failed",
                provider=voice_config.stt_provider,
                exc_info=True,
            )

    # Fall back to local
    local_provider = getattr(app_state, "stt_provider", None)
    logger.info(
        "stt_resolve_local_fallback",
        local_available=local_provider is not None,
        local_provider=getattr(local_provider, "provider_name", None),
    )
    return local_provider


def _resolve_tts_for_tenant(
    voice_config: Any,
    app_state: Any,
) -> tuple[Any, int]:
    """Resolve the TTS provider + sample rate for a tenant.

    Returns:
        Tuple of (tts_provider, sample_rate).
        Azure outputs 16kHz; Piper outputs 22050Hz.
    """
    logger.debug(
        "tts_resolve_start",
        configured_provider=voice_config.tts_provider,
        has_api_key=bool(voice_config.tts_api_key),
        azure_region=getattr(voice_config, "azure_region", None),
        tts_voice=voice_config.tts_voice,
    )

    if (
        voice_config.tts_provider
        and voice_config.tts_provider != "piper"
        and voice_config.tts_api_key
    ):
        from app.infrastructure.tts.factory import get_tts_provider

        logger.info(
            "tts_resolve_cloud",
            provider=voice_config.tts_provider,
            region=getattr(voice_config, "azure_region", "eastus"),
            voice=voice_config.tts_voice,
        )
        try:
            provider = get_tts_provider(
                voice_config.tts_provider,
                subscription_key=voice_config.tts_api_key,
                region=getattr(voice_config, "azure_region", "eastus"),
                voice=voice_config.tts_voice,
            )
            # Azure Raw16Khz16BitMonoPcm → 16000
            sample_rate = 16000 if voice_config.tts_provider == "azure" else 22050
            logger.info(
                "tts_resolve_cloud_ok",
                provider=provider.provider_name,
                sample_rate=sample_rate,
                voice=voice_config.tts_voice,
            )
            return provider, sample_rate
        except Exception:
            logger.warning(
                "tenant_cloud_tts_init_failed",
                provider=voice_config.tts_provider,
                exc_info=True,
            )

    # Fall back to local
    local_provider = getattr(app_state, "tts_provider", None)
    logger.info(
        "tts_resolve_local_fallback",
        local_available=local_provider is not None,
        local_provider=getattr(local_provider, "provider_name", None),
        sample_rate=22050,
    )
    return local_provider, 22050

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
        "azure_region": voice_config.azure_region,
        "has_api_key": bool(voice_config.stt_api_key),
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

    settings = get_settings()

    voice_config = resolve_tenant_voice_config(
        config,
        encryption_key=getattr(settings, "secret_key", None),
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
        "azure_region": voice_config.azure_region,
        "has_api_key": bool(voice_config.stt_api_key),
    }


@router.post("/transcribe")
async def transcribe_audio(
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> dict:
    """Transcribe an uploaded audio file to text via STT.

    Accepts multipart form upload with an ``audio`` field, or raw body.
    Audio is decoded via PyAV (webm, wav, mp3, ogg, etc) to PCM,
    then transcribed by the tenant's configured STT provider.
    """
    # Resolve per-tenant STT provider
    settings = get_settings()
    tenant_repo = SQLTenantRepository(session)
    tenant = await tenant_repo.get_by_id(user.tenant_id)
    tenant_config_json = (tenant.config_json or {}) if tenant else {}

    stt_locally = getattr(request.app.state, "stt_provider", None) is not None
    tts_locally = getattr(request.app.state, "tts_provider", None) is not None

    voice_config = resolve_tenant_voice_config(
        tenant_config_json,
        encryption_key=getattr(settings, "secret_key", None),
        stt_locally_available=stt_locally,
        tts_locally_available=tts_locally,
    )

    stt_provider = _resolve_stt_for_tenant(voice_config, request.app.state)
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
    else:
        # Raw body upload
        audio_bytes = await request.body()

    if not audio_bytes:
        logger.warning("transcribe_empty_audio", tenant_id=user.tenant_id)
        return {"error": "Empty audio data", "text": ""}

    logger.info(
        "transcribe_audio_received",
        tenant_id=user.tenant_id,
        audio_bytes=len(audio_bytes),
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

        # Resample to 16kHz mono Int16 PCM (common format for STT providers)
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

        logger.info(
            "transcribe_audio_decoded",
            pcm_bytes=len(pcm_bytes),
            duration_secs=round(len(pcm_bytes) / (16000 * 2), 2),
            provider=stt_provider.provider_name,
        )

        t0 = time.monotonic()

        # ── Provider-specific transcription ──────────────────────
        if stt_provider.provider_name == "whisper":
            # Whisper requires float32 numpy array in [-1.0, 1.0].
            # Must consume the segment generator inside the thread
            # (CT2 model state is NOT thread-safe).
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            def _transcribe_sync() -> tuple[str, object]:
                segments, info = stt_provider._model.transcribe(samples, language="en")
                text = " ".join(s.text.strip() for s in segments)
                return text, info

            text, info = await asyncio.to_thread(_transcribe_sync)
            language = getattr(info, "language", "en")
            lang_prob = round(getattr(info, "language_probability", 0), 3)
        else:
            # Cloud providers (Azure, etc.) accept raw PCM Int16 bytes
            # via the STTProvider.transcribe() interface.
            text = await stt_provider.transcribe(
                pcm_bytes,
                sample_rate=16000,
                language="en",
            )
            language = "en"
            lang_prob = 1.0

        elapsed = round(time.monotonic() - t0, 3)

        logger.info(
            "transcribe_completed",
            tenant_id=user.tenant_id,
            text_length=len(text),
            text_preview=text[:100] if text else "(empty)",
            elapsed_secs=elapsed,
            provider=stt_provider.provider_name,
            language=language,
            language_probability=lang_prob,
        )

        return {"text": text, "language": language}

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
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    """Synthesize text to speech using TTS provider.

    Accepts JSON ``{"text": "..."}`` and returns WAV audio.
    """
    # Resolve per-tenant TTS provider
    settings = get_settings()
    tenant_repo = SQLTenantRepository(session)
    tenant = await tenant_repo.get_by_id(user.tenant_id)
    tenant_config_json = (tenant.config_json or {}) if tenant else {}

    stt_locally = getattr(request.app.state, "stt_provider", None) is not None
    tts_locally = getattr(request.app.state, "tts_provider", None) is not None

    voice_config = resolve_tenant_voice_config(
        tenant_config_json,
        encryption_key=getattr(settings, "secret_key", None),
        stt_locally_available=stt_locally,
        tts_locally_available=tts_locally,
    )

    tts_provider, tts_sample_rate = _resolve_tts_for_tenant(voice_config, request.app.state)
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
        provider=tts_provider.provider_name,
        sample_rate=tts_sample_rate,
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
            provider=tts_provider.provider_name,
        )

        # Wrap raw PCM in a WAV header for browser playback
        # Azure outputs 16kHz; Piper outputs 22050Hz
        sample_rate = tts_sample_rate
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


@router.put("/config")
async def save_voice_config(
    request: Request,
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> dict:
    """Save voice provider configuration for this tenant (admin only).

    Body (all optional):
        stt_provider: ``"whisper"`` | ``"azure"``
        tts_provider: ``"piper"`` | ``"azure"``
        api_key: API key for cloud provider (Azure subscription key)
        azure_region: Azure region (e.g. ``eastus``)
        tts_voice: Voice name (e.g. ``en-US-AriaNeural``)
    """
    from app.core.crypto import encrypt_value, mask_api_key
    from app.infrastructure.database.connection import AsyncSessionLocal

    body = await request.json()

    logger.info(
        "voice_config_save_requested",
        tenant_id=user.tenant_id,
        user_id=user.id,
        stt_provider=body.get("stt_provider"),
        tts_provider=body.get("tts_provider"),
        has_api_key=bool(body.get("api_key")),
        azure_region=body.get("azure_region"),
        tts_voice=body.get("tts_voice"),
    )

    settings = get_settings()

    async with AsyncSessionLocal() as session:
        repo = SQLTenantRepository(session)
        tenant = await repo.get_by_id(user.tenant_id)
        if not tenant:
            logger.warning(
                "voice_config_save_tenant_not_found",
                tenant_id=user.tenant_id,
            )
            return {"error": "Tenant not found"}

        config = dict(tenant.config_json or {})
        logger.debug(
            "voice_config_save_existing",
            tenant_id=user.tenant_id,
            existing_stt=config.get("stt_provider"),
            existing_tts=config.get("tts_provider"),
            existing_has_key=bool(config.get("stt_api_key")),
        )

        # Update provider names
        stt_provider = body.get("stt_provider")
        tts_provider = body.get("tts_provider")

        if stt_provider is not None:
            config["stt_provider"] = stt_provider if stt_provider else None
        if tts_provider is not None:
            config["tts_provider"] = tts_provider if tts_provider else None

        # Update API key (encrypted at rest)
        api_key = body.get("api_key")
        if api_key:
            encryption_key = getattr(settings, "secret_key", None)
            encrypted = encrypt_value(api_key, encryption_key) if encryption_key else api_key
            # Azure uses same key for both STT and TTS
            config["stt_api_key"] = encrypted
            config["tts_api_key"] = encrypted
            logger.info(
                "voice_config_api_key_encrypted",
                tenant_id=user.tenant_id,
                key_length=len(api_key),
                encrypted_length=len(encrypted),
                used_encryption=bool(encryption_key),
            )

        # Update Azure region
        azure_region = body.get("azure_region")
        if azure_region:
            config["azure_speech_region"] = azure_region

        # Update TTS voice
        tts_voice = body.get("tts_voice")
        if tts_voice:
            config["tts_voice"] = tts_voice

        await repo.update(user.tenant_id, config_json=config)
        await session.commit()

        logger.info(
            "voice_config_persisted",
            tenant_id=user.tenant_id,
            stt_provider=config.get("stt_provider"),
            tts_provider=config.get("tts_provider"),
            azure_region=config.get("azure_speech_region"),
            tts_voice=config.get("tts_voice"),
            has_api_key=bool(config.get("stt_api_key")),
        )

    # Re-resolve and return current config
    stt_locally = getattr(request.app.state, "stt_provider", None) is not None
    tts_locally = getattr(request.app.state, "tts_provider", None) is not None

    voice_config = resolve_tenant_voice_config(
        config,
        encryption_key=getattr(settings, "secret_key", None),
        stt_locally_available=stt_locally,
        tts_locally_available=tts_locally,
    )

    response = {
        "voice_enabled": voice_config.voice_enabled,
        "stt_provider": voice_config.stt_provider,
        "tts_provider": voice_config.tts_provider,
        "tts_voice": voice_config.tts_voice,
        "azure_region": voice_config.azure_region,
        "has_api_key": bool(voice_config.stt_api_key),
        "api_key_preview": mask_api_key(api_key) if api_key else None,
    }

    logger.info(
        "voice_config_save_response",
        tenant_id=user.tenant_id,
        voice_enabled=response["voice_enabled"],
        stt_provider=response["stt_provider"],
        tts_provider=response["tts_provider"],
        has_api_key=response["has_api_key"],
    )

    return response
