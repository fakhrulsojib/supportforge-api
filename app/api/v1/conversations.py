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
    """
    limit = min(limit, 100)
    repo = SQLConversationRepository(session)
    conversations = await repo.list_by_tenant(user.tenant_id, limit=limit, offset=offset)

    return ConversationListResponse(
        conversations=[
            ConversationSummaryResponse(
                id=c.id,
                tenant_id=c.tenant_id,
                user_id=c.user_id,
                status=c.status,
                started_at=c.started_at,
            )
            for c in conversations
        ],
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
    from app.core.exceptions import ConversationNotFoundError

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

    Args:
        message_id: Message UUID.
        request: Feedback data.
        session: Database session.
        user: Authenticated user.

    Returns:
        Updated message with feedback.
    """
    from app.core.exceptions import SupportForgeError

    msg_repo = SQLMessageRepository(session)
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
