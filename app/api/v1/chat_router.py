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
    # ── Tenant status gate ───────────────────────────────────────
    # Only ACTIVE tenants can access chat. Pending, suspended, and
    # archived tenants are all blocked.
    tenant_repo = SQLTenantRepository(session)
    tenant = await tenant_repo.get_by_id(user.tenant_id)
    if not tenant or tenant.status != TenantStatus.ACTIVE:
        raise TenantSuspendedError(tenant_id=user.tenant_id)

    # Read per-tenant model overrides from config_json
    tenant_chat_model: str | None = None
    tenant_embedding_model: str | None = None
    if tenant.config_json:
        raw_chat = tenant.config_json.get("chat_model")
        if isinstance(raw_chat, str) and raw_chat:
            tenant_chat_model = raw_chat
        raw_embed = tenant.config_json.get("embedding_model")
        if isinstance(raw_embed, str) and raw_embed:
            tenant_embedding_model = raw_embed

    result = await chat_service.process_message(
        message=request.message,
        tenant_id=user.tenant_id,
        conversation_id=request.conversation_id,
        tenant_chat_model=tenant_chat_model,
        tenant_embedding_model=tenant_embedding_model,
    )

    sources = [
        SourceCitation(
            content=s.get("content", ""),
            score=s.get("score", 0.0),
            id=s.get("id", ""),
        )
        for s in result.get("sources", [])
    ]

    return ChatResponse(
        answer=result["answer"],
        conversation_id=result["conversation_id"],
        sources=sources,
        escalated=result.get("escalated", False),
        escalation_reason=result.get("escalation_reason", ""),
        escalation_trigger=result.get("escalation_trigger", "none"),
        model_used=result.get("model_used", ""),
    )
