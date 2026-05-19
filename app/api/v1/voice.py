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
