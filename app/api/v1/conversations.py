"""Conversation management API router — CRUD endpoints.

Provides listing, retrieval, and feedback endpoints for conversations.
All endpoints are scoped to the authenticated user's tenant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends

from app.api.schemas.conversation import (
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationSummaryResponse,
    FeedbackRequest,
    MessageResponse,
)
from app.core.dependencies import get_current_user
from app.core.exceptions import ConversationNotFoundError, SupportForgeError
from app.infrastructure.database.connection import get_async_session
from app.infrastructure.database.repositories.conversation_repo import (
    SQLConversationRepository,
    SQLMessageRepository,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.domain.models.user import User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


@router.get("/", response_model=ConversationListResponse)
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_user),
) -> ConversationListResponse:
    """List conversations for the authenticated user's tenant.

    Args:
        limit: Page size (max 100).
        offset: Page offset.
        session: Database session.
        user: Authenticated user.

    Returns:
        Paginated list of conversation summaries.

    Note:
        M6: ``total`` reflects the current page length, not global count.
        A separate COUNT query will be added when real pagination is needed
        in Phase 3 frontend integration. For now the field is accurate for
        single-page results.
    """
    limit = min(limit, 100)
    repo = SQLConversationRepository(session)
    msg_repo = SQLMessageRepository(session)
    conversations = await repo.list_by_tenant(user.tenant_id, limit=limit, offset=offset)

    summaries: list[ConversationSummaryResponse] = []
    for c in conversations:
        # Get the first message as the conversation title
        messages = await msg_repo.list_by_conversation(c.id, limit=1)
        title = ""
        if messages:
            # Truncate to 60 chars for sidebar display
            raw = messages[0].content.strip()
            title = raw[:60] + ("…" if len(raw) > 60 else "")

        summaries.append(
            ConversationSummaryResponse(
                id=c.id,
                tenant_id=c.tenant_id,
                user_id=c.user_id,
                status=c.status,
                title=title,
                started_at=c.started_at,
            )
        )

    return ConversationListResponse(
        conversations=summaries,
        total=len(conversations),
        limit=limit,
        offset=offset,
    )


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: str,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_user),
) -> ConversationDetailResponse:
    """Get a full conversation with all messages.

    Args:
        conversation_id: Conversation UUID.
        session: Database session.
        user: Authenticated user.

    Returns:
        Full conversation with messages.
    """
    conv_repo = SQLConversationRepository(session)
    msg_repo = SQLMessageRepository(session)

    conversation = await conv_repo.get_by_id(conversation_id)
    if not conversation:
        raise ConversationNotFoundError(conversation_id=conversation_id)

    # Verify tenant isolation
    if conversation.tenant_id != user.tenant_id:
        raise ConversationNotFoundError(conversation_id=conversation_id)

    messages = await msg_repo.list_by_conversation(conversation_id)

    return ConversationDetailResponse(
        id=conversation.id,
        tenant_id=conversation.tenant_id,
        user_id=conversation.user_id,
        status=conversation.status,
        messages=[
            MessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                sources_json=m.sources_json,
                model_used=m.model_used,
                tokens_in=m.tokens_in,
                tokens_out=m.tokens_out,
                feedback=m.feedback,
                created_at=m.created_at,
            )
            for m in messages
        ],
        started_at=conversation.started_at,
    )


@router.patch("/messages/{message_id}/feedback", response_model=MessageResponse)
async def update_message_feedback(
    message_id: str,
    request: FeedbackRequest,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_user),
) -> MessageResponse:
    """Update feedback on a specific message.

    M1: Verifies the message belongs to a conversation owned by the
    user's tenant BEFORE performing the update.

    Args:
        message_id: Message UUID.
        request: Feedback data.
        session: Database session.
        user: Authenticated user.

    Returns:
        Updated message with feedback.
    """
    msg_repo = SQLMessageRepository(session)
    conv_repo = SQLConversationRepository(session)

    # M1: Verify tenant ownership BEFORE updating
    existing = await msg_repo.get_by_id(message_id)
    if not existing:
        raise SupportForgeError(
            message=f"Message '{message_id}' not found",
            status_code=404,
            error_code="MESSAGE_NOT_FOUND",
        )

    conversation = await conv_repo.get_by_id(existing.conversation_id)
    if not conversation or conversation.tenant_id != user.tenant_id:
        # Return 404 to prevent cross-tenant message existence leakage
        raise SupportForgeError(
            message=f"Message '{message_id}' not found",
            status_code=404,
            error_code="MESSAGE_NOT_FOUND",
        )

    # Tenant verified — safe to update
    message = await msg_repo.update_feedback(message_id, request.feedback)
    if not message:
        raise SupportForgeError(
            message=f"Message '{message_id}' not found",
            status_code=404,
            error_code="MESSAGE_NOT_FOUND",
        )

    logger.info("feedback_updated", message_id=message_id, feedback=request.feedback.value)

    return MessageResponse(
        id=message.id,
        role=message.role,
        content=message.content,
        sources_json=message.sources_json,
        model_used=message.model_used,
        tokens_in=message.tokens_in,
        tokens_out=message.tokens_out,
        feedback=message.feedback,
        created_at=message.created_at,
    )
