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

if TYPE_CHECKING:
    from app.domain.models.user import User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    chat_service: Any = Depends(get_chat_service),
) -> ChatResponse:
    """Process a chat message through the RAG pipeline.

    Requires JWT authentication. Tenant is derived from the
    authenticated user's token — no manual header needed.

    Args:
        request: Chat request with message and optional conversation_id.
        user: Authenticated user (injected via JWT).
        chat_service: ChatService singleton (injected via app.state).

    Returns:
        ChatResponse with AI-generated answer and source citations.
    """
    result = await chat_service.process_message(
        message=request.message,
        tenant_id=user.tenant_id,
        conversation_id=request.conversation_id,
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
