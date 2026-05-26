"""Chat API router — POST /api/v1/chat.

Provides the REST endpoint for synchronous chat requests. For real-time
streaming, use the WebSocket endpoint in ``chat_ws.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from fastapi import APIRouter, Depends

from app.api.schemas.chat import ChatRequest, ChatResponse, SourceCitation
from app.core.dependencies import get_chat_service, get_current_user
from app.core.exceptions import TenantSuspendedError
from app.domain.models.enums import TenantStatus
from app.infrastructure.database.connection import get_async_session
from app.infrastructure.database.repositories.tenant_repo import SQLTenantRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.domain.models.user import User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    chat_service: Any = Depends(get_chat_service),
    session: AsyncSession = Depends(get_async_session),
) -> ChatResponse:
    """Process a chat message through the RAG pipeline.

    Requires JWT authentication. Tenant is derived from the
    authenticated user's token — no manual header needed.

    Non-active tenants are rejected with a 403 error.

    Args:
        request: Chat request with message and optional conversation_id.
        user: Authenticated user (injected via JWT).
        chat_service: ChatService singleton (injected via app.state).
        session: Database session for tenant status check.

    Returns:
        ChatResponse with AI-generated answer and source citations.
    """
    logger.debug(
        "incoming_chat_request",
        tenant_id=user.tenant_id,
        conversation_id=request.conversation_id,
        message_length=len(request.message)
    )

    # ── Tenant status gate ───────────────────────────────────────
    # Only ACTIVE tenants can access chat. Pending, suspended, and
    # archived tenants are all blocked.
    tenant_repo = SQLTenantRepository(session)
    tenant = await tenant_repo.get_by_id(user.tenant_id)
    if not tenant or tenant.status != TenantStatus.ACTIVE:
        logger.warning(
            "chat_request_rejected",
            reason="tenant_suspended_or_not_found",
            tenant_id=user.tenant_id
        )
        raise TenantSuspendedError(tenant_id=user.tenant_id)

    # Read per-tenant model overrides from config_json
    from app.config import get_settings
    from app.core.tenant_config import resolve_tenant_models
    settings = get_settings()

    # Load tenant secrets for API key resolution (secrets > config_json)
    from app.infrastructure.database.repositories.tenant_secret_repo import (
        SQLTenantSecretRepository,
    )
    sec_repo = SQLTenantSecretRepository(session, encryption_key=settings.secret_key)
    try:
        tenant_secrets = await sec_repo.get_all_decrypted(user.tenant_id)
    except Exception:
        tenant_secrets = {}

    models = resolve_tenant_models(
        tenant.config_json,
        encryption_key=settings.secret_key,
        secrets=tenant_secrets,
    )

    # Extract agent personality config (no decryption — plain dict)
    raw_agent_config = tenant.config_json.get("agent_prompt") if tenant.config_json else None
    tenant_agent_config = raw_agent_config if isinstance(raw_agent_config, dict) else None

    result = await chat_service.process_message(
        message=request.message,
        tenant_id=user.tenant_id,
        conversation_id=request.conversation_id,
        tenant_chat_model=models.chat_model,
        tenant_embedding_model=models.embedding_model,
        tenant_chat_provider=models.chat_provider,
        tenant_gemini_api_key=models.gemini_api_key,
        tenant_embedding_provider=models.embedding_provider,
        tenant_gemini_embedding_api_key=models.gemini_embedding_api_key,
        tenant_agent_config=tenant_agent_config,
        tenant_config_json=tenant.config_json,
    )

    sources = [
        SourceCitation(
            content=s.get("content", ""),
            score=s.get("score", 0.0),
            id=s.get("id", ""),
        )
        for s in result.get("sources", [])
    ]

    logger.info(
        "chat_request_completed",
        conversation_id=result["conversation_id"],
        model_used=result.get("model_used", ""),
        sources_count=len(sources),
        escalated=result.get("escalated", False)
    )

    return ChatResponse(
        answer=result["answer"],
        conversation_id=result["conversation_id"],
        sources=sources,
        escalated=result.get("escalated", False),
        escalation_reason=result.get("escalation_reason", ""),
        escalation_trigger=result.get("escalation_trigger", "none"),
        model_used=result.get("model_used", ""),
    )
