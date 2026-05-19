"""Voice API endpoints — configuration, health, and session info.

Provides REST endpoints for:
- Checking voice availability for a tenant
- Voice service health status
- Active voice session counts (admin only)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

router = APIRouter(prefix="/api/v1/voice", tags=["Voice"])


@router.get("/config")
async def get_voice_config(
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> dict:
    """Return voice availability for the authenticated tenant.

    Resolves the three-tier voice configuration:
    1. Cloud STT/TTS with API keys
    2. Local whisper/piper if available
    3. Disabled
    """
    # Fetch tenant config_json from the database
    tenant_repo = SQLTenantRepository(session)
    tenant = await tenant_repo.get_by_id(user.tenant_id)
    tenant_config_json: dict = {}
    if tenant and tenant.config_json:
        tenant_config_json = tenant.config_json

    settings = get_settings()

    stt_available = hasattr(request.app.state, "stt_provider") and request.app.state.stt_provider is not None
    tts_available = hasattr(request.app.state, "tts_provider") and request.app.state.tts_provider is not None

    voice_config = resolve_tenant_voice_config(
        tenant_config_json,
        encryption_key=settings.secret_key,
        stt_locally_available=stt_available,
        tts_locally_available=tts_available,
    )

    return {
        "voice_enabled": voice_config.voice_enabled,
        "stt_provider": voice_config.stt_provider,
        "tts_provider": voice_config.tts_provider,
        "tts_voice": voice_config.tts_voice,
        "max_voice_sessions": voice_config.max_voice_sessions,
    }


@router.get("/health")
async def get_voice_health(
    request: Request,
    user: User = Depends(get_current_user),
) -> dict:
    """Return health status of STT and TTS services."""
    stt_available = False
    tts_available = False

    stt_provider = getattr(request.app.state, "stt_provider", None)
    if stt_provider is not None:
        try:
            stt_available = await stt_provider.health_check()
        except Exception:
            pass

    tts_provider = getattr(request.app.state, "tts_provider", None)
    if tts_provider is not None:
        try:
            tts_available = await tts_provider.health_check()
        except Exception:
            pass

    return {
        "stt_available": stt_available,
        "tts_available": tts_available,
        "stt_provider": getattr(stt_provider, "provider_name", None) if stt_provider else None,
        "tts_provider": getattr(tts_provider, "provider_name", None) if tts_provider else None,
    }


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

    return {
        "tenant_id": user.tenant_id,
        "active_sessions": active,
    }
