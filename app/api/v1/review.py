"""Review Queue API router — admin-only endpoints for feedback review.

Provides listing, filtering, and action endpoints for:
- Negatively-rated messages
- Escalated conversations
- Flagged messages (output validation failures)

All endpoints enforce admin-only access via ``require_role(UserRole.ADMIN)``.
Tenant isolation is enforced through the authenticated user's tenant_id.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends, Query

from app.api.schemas.review import (
    EscalationItemResponse,
    EscalationListResponse,
    EscalationReviewActionResponse,
    FlaggedMessageListResponse,
    NegativeFeedbackListResponse,
    ReviewActionResponse,
    ReviewItemResponse,
    ReviewStatsResponse,
)
from app.core.dependencies import require_role
from app.core.exceptions import SupportForgeError
from app.domain.models.enums import EscalationTrigger, UserRole
from app.infrastructure.database.connection import get_async_session
from app.infrastructure.database.repositories.conversation_repo import (
    SQLConversationRepository,
    SQLMessageRepository,
)
from app.infrastructure.database.repositories.user_repo import SQLUserRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.domain.models.user import User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["review"])


@router.get("/feedback/negative", response_model=NegativeFeedbackListResponse)
async def list_negative_feedback(
    reviewed: bool | None = Query(None, description="Filter: True=reviewed, False=unreviewed, None=all"),
    start_date: str | None = Query(None, description="ISO date lower bound"),
    end_date: str | None = Query(None, description="ISO date upper bound"),
    limit: int = Query(50, ge=1, le=100, description="Page size"),
    offset: int = Query(0, ge=0, description="Page offset"),
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> NegativeFeedbackListResponse:
    """List messages with negative feedback for the admin's tenant.

    Includes the preceding user question for context.

    Args:
        reviewed: Optional reviewed/unreviewed filter.
        start_date: Optional ISO date lower bound.
        end_date: Optional ISO date upper bound.
        limit: Page size (max 100).
        offset: Page offset.
        session: Database session.
        user: Authenticated admin user.

    Returns:
        Paginated list of negative feedback items.
    """
    logger.debug("api_review_list_negative", tenant_id=user.tenant_id, reviewed=reviewed, limit=limit, offset=offset)
    msg_repo = SQLMessageRepository(session)
    conv_repo = SQLConversationRepository(session)
    user_repo = SQLUserRepository(session)

    messages, total = await msg_repo.list_negative_feedback(
        tenant_id=user.tenant_id,
        reviewed=reviewed,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )

    # Batch-fetch preceding user questions (avoids N+1)
    msg_ids = [m.id for m in messages]
    preceding_map = await msg_repo.get_preceding_user_messages_batch(msg_ids)

    # Batch-resolve user emails from conversation owners
    conv_ids = list({m.conversation_id for m in messages})
    email_map: dict[str, str] = {}
    for cid in conv_ids:
        conv = await conv_repo.get_by_id(cid)
        if conv and conv.user_id:
            owner = await user_repo.get_by_id(conv.user_id)
            if owner:
                email_map[cid] = owner.email

    items: list[ReviewItemResponse] = []
    for msg in messages:
        preceding = preceding_map.get(msg.id)
        user_question = preceding.content if preceding else ""

        items.append(
            ReviewItemResponse(
                message_id=msg.id,
                conversation_id=msg.conversation_id,
                user_email=email_map.get(msg.conversation_id, ""),
                user_question=user_question,
                ai_answer=msg.content,
                sources_json=msg.sources_json,
                feedback=msg.feedback,
                validation_status=msg.validation_status,
                moderation_reason=msg.moderation_reason,
                moderation_matched_term=msg.moderation_matched_term,
                reviewed_at=msg.reviewed_at,
                reviewed_by=msg.reviewed_by,
                created_at=msg.created_at,
            )
        )

    return NegativeFeedbackListResponse(
        items=items, total=total, limit=limit, offset=offset,
    )


@router.get("/escalations", response_model=EscalationListResponse)
async def list_escalations(
    trigger: str | None = Query(None, description="Filter by trigger type"),
    reviewed: bool | None = Query(None, description="Filter: True=reviewed, False=unreviewed, None=all"),
    start_date: str | None = Query(None, description="ISO date lower bound"),
    end_date: str | None = Query(None, description="ISO date upper bound"),
    limit: int = Query(50, ge=1, le=100, description="Page size"),
    offset: int = Query(0, ge=0, description="Page offset"),
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> EscalationListResponse:
    """List escalated conversations for the admin's tenant.

    Includes the first user message as a preview.

    Args:
        trigger: Optional escalation trigger type filter.
        reviewed: Optional reviewed/unreviewed filter.
        start_date: Optional ISO date lower bound.
        end_date: Optional ISO date upper bound.
        limit: Page size (max 100).
        offset: Page offset.
        session: Database session.
        user: Authenticated admin user.

    Returns:
        Paginated list of escalated conversations.
    """
    logger.debug("api_review_list_escalations", tenant_id=user.tenant_id, trigger=trigger, reviewed=reviewed, limit=limit)
    conv_repo = SQLConversationRepository(session)
    msg_repo = SQLMessageRepository(session)
    user_repo = SQLUserRepository(session)

    # Parse trigger filter
    trigger_enum: EscalationTrigger | None = None
    if trigger:
        with contextlib.suppress(ValueError):
            trigger_enum = EscalationTrigger(trigger)

    conversations, total = await conv_repo.list_escalated(
        tenant_id=user.tenant_id,
        trigger=trigger_enum,
        reviewed=reviewed,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )

    items: list[EscalationItemResponse] = []
    for conv in conversations:
        # Get first message as preview
        first_message = ""
        messages = await msg_repo.list_by_conversation(conv.id, limit=1)
        if messages:
            first_message = messages[0].content[:200]

        # Resolve owner email
        owner_email = ""
        if conv.user_id:
            owner = await user_repo.get_by_id(conv.user_id)
            if owner:
                owner_email = owner.email

        items.append(
            EscalationItemResponse(
                conversation_id=conv.id,
                user_email=owner_email,
                trigger=conv.escalation_trigger,
                first_message=first_message,
                status=conv.status,
                reviewed_at=conv.reviewed_at,
                reviewed_by=conv.reviewed_by,
                started_at=conv.started_at,
            )
        )

    return EscalationListResponse(
        items=items, total=total, limit=limit, offset=offset,
    )


@router.patch("/escalations/{conversation_id}/review", response_model=EscalationReviewActionResponse)
async def mark_escalation_reviewed(
    conversation_id: str,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> EscalationReviewActionResponse:
    """Mark an escalated conversation as reviewed.

    Sets ``reviewed_at`` to current UTC time and ``reviewed_by`` to
    the admin's user ID. Verifies tenant ownership before updating.

    Args:
        conversation_id: Conversation UUID to mark as reviewed.
        session: Database session.
        user: Authenticated admin user.

    Returns:
        Review action confirmation with timestamp.

    Raises:
        SupportForgeError: If conversation not found or belongs to different tenant.
    """
    logger.debug("api_review_mark_escalation_reviewed", conversation_id=conversation_id, user_id=user.id)
    conv_repo = SQLConversationRepository(session)

    # Verify conversation exists and belongs to admin's tenant
    existing = await conv_repo.get_by_id(conversation_id)
    if not existing or existing.tenant_id != user.tenant_id:
        raise SupportForgeError(
            message=f"Conversation '{conversation_id}' not found",
            status_code=404,
            error_code="CONVERSATION_NOT_FOUND",
        )

    # Mark as reviewed
    updated = await conv_repo.update_escalation_review_status(conversation_id, user.id)
    if not updated:
        raise SupportForgeError(
            message=f"Conversation '{conversation_id}' not found",
            status_code=404,
            error_code="CONVERSATION_NOT_FOUND",
        )

    await session.commit()

    logger.info(
        "escalation_marked_reviewed",
        conversation_id=conversation_id,
        reviewed_by=user.id,
    )

    return EscalationReviewActionResponse(
        conversation_id=updated.id,
        reviewed_at=updated.reviewed_at,  # type: ignore[arg-type]
        reviewed_by=updated.reviewed_by,
    )


@router.get("/flagged", response_model=FlaggedMessageListResponse)
async def list_flagged_messages(
    reviewed: bool | None = Query(None, description="Filter: True=reviewed, False=unreviewed, None=all"),
    start_date: str | None = Query(None, description="ISO date lower bound"),
    end_date: str | None = Query(None, description="ISO date upper bound"),
    limit: int = Query(50, ge=1, le=100, description="Page size"),
    offset: int = Query(0, ge=0, description="Page offset"),
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> FlaggedMessageListResponse:
    """List flagged messages (output validation failures) for the admin's tenant.

    Args:
        reviewed: Optional reviewed/unreviewed filter.
        start_date: Optional ISO date lower bound.
        end_date: Optional ISO date upper bound.
        limit: Page size (max 100).
        offset: Page offset.
        session: Database session.
        user: Authenticated admin user.

    Returns:
        Paginated list of flagged messages.
    """
    logger.debug("api_review_list_flagged", tenant_id=user.tenant_id, reviewed=reviewed, limit=limit, offset=offset)
    msg_repo = SQLMessageRepository(session)
    conv_repo = SQLConversationRepository(session)
    user_repo = SQLUserRepository(session)

    messages, total = await msg_repo.list_flagged_messages(
        tenant_id=user.tenant_id,
        reviewed=reviewed,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )

    # Batch-fetch preceding user questions (avoids N+1)
    msg_ids = [m.id for m in messages]
    preceding_map = await msg_repo.get_preceding_user_messages_batch(msg_ids)

    # Batch-resolve user emails from conversation owners
    conv_ids = list({m.conversation_id for m in messages})
    email_map: dict[str, str] = {}
    for cid in conv_ids:
        conv = await conv_repo.get_by_id(cid)
        if conv and conv.user_id:
            owner = await user_repo.get_by_id(conv.user_id)
            if owner:
                email_map[cid] = owner.email

    items: list[ReviewItemResponse] = []
    for msg in messages:
        preceding = preceding_map.get(msg.id)
        user_question = preceding.content if preceding else ""

        items.append(
            ReviewItemResponse(
                message_id=msg.id,
                conversation_id=msg.conversation_id,
                user_email=email_map.get(msg.conversation_id, ""),
                user_question=user_question,
                ai_answer=msg.content,
                sources_json=msg.sources_json,
                feedback=msg.feedback,
                validation_status=msg.validation_status,
                moderation_reason=msg.moderation_reason,
                moderation_matched_term=msg.moderation_matched_term,
                reviewed_at=msg.reviewed_at,
                reviewed_by=msg.reviewed_by,
                created_at=msg.created_at,
            )
        )

    return FlaggedMessageListResponse(
        items=items, total=total, limit=limit, offset=offset,
    )


@router.patch("/feedback/{message_id}/review", response_model=ReviewActionResponse)
async def mark_reviewed(
    message_id: str,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> ReviewActionResponse:
    """Mark a feedback item as reviewed.

    Sets ``reviewed_at`` to current UTC time and ``reviewed_by`` to
    the admin's user ID. Verifies tenant ownership before updating.

    Args:
        message_id: Message UUID to mark as reviewed.
        session: Database session.
        user: Authenticated admin user.

    Returns:
        Review action confirmation with timestamp.

    Raises:
        SupportForgeError: If message not found or belongs to different tenant.
    """
    logger.debug("api_review_mark_feedback_reviewed", message_id=message_id, user_id=user.id)
    msg_repo = SQLMessageRepository(session)
    conv_repo = SQLConversationRepository(session)

    # Verify message exists
    existing = await msg_repo.get_by_id(message_id)
    if not existing:
        raise SupportForgeError(
            message=f"Message '{message_id}' not found",
            status_code=404,
            error_code="MESSAGE_NOT_FOUND",
        )

    # Verify tenant ownership through conversation
    conversation = await conv_repo.get_by_id(existing.conversation_id)
    if not conversation or conversation.tenant_id != user.tenant_id:
        raise SupportForgeError(
            message=f"Message '{message_id}' not found",
            status_code=404,
            error_code="MESSAGE_NOT_FOUND",
        )

    # Mark as reviewed
    updated = await msg_repo.update_review_status(message_id, user.id)
    if not updated:
        raise SupportForgeError(
            message=f"Message '{message_id}' not found",
            status_code=404,
            error_code="MESSAGE_NOT_FOUND",
        )

    await session.commit()

    logger.info(
        "feedback_marked_reviewed",
        message_id=message_id,
        reviewed_by=user.id,
    )

    return ReviewActionResponse(
        message_id=updated.id,
        reviewed_at=updated.reviewed_at,  # type: ignore[arg-type]
        reviewed_by=updated.reviewed_by,
    )


@router.get("/feedback/stats", response_model=ReviewStatsResponse)
async def get_review_stats(
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> ReviewStatsResponse:
    """Get aggregate review queue counts for badge display.

    Returns unreviewed negative feedback count, unreviewed flagged count,
    open escalation count, and unresolved failed queries count — all
    scoped to the admin's tenant.

    Args:
        session: Database session.
        user: Authenticated admin user.

    Returns:
        Aggregate review queue statistics.
    """
    logger.debug("api_review_stats", tenant_id=user.tenant_id)
    from app.infrastructure.database.repositories.failed_query_repo import (
        SQLFailedQueryRepository,
    )

    msg_repo = SQLMessageRepository(session)
    conv_repo = SQLConversationRepository(session)
    fq_repo = SQLFailedQueryRepository(session)

    unreviewed_negative = await msg_repo.count_unreviewed_negative(user.tenant_id)
    unreviewed_flagged = await msg_repo.count_unreviewed_flagged(user.tenant_id)
    open_escalations = await conv_repo.count_open_escalations(user.tenant_id)
    unresolved_fqs = await fq_repo.count_unresolved(user.tenant_id)

    return ReviewStatsResponse(
        unreviewed_negative=unreviewed_negative,
        unreviewed_flagged=unreviewed_flagged,
        open_escalations=open_escalations,
        unresolved_failed_queries=unresolved_fqs,
    )
