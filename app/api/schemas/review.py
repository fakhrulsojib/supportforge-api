"""Review Queue API schemas — request/response DTOs.

Admin-only endpoints for reviewing negative feedback, escalations,
and flagged messages. Used by ``app.api.v1.review`` router.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003 — Pydantic needs runtime access

from pydantic import BaseModel, Field

from app.domain.models.enums import (  # noqa: TCH001 — Pydantic runtime
    ConversationStatus,
    EscalationTrigger,
    FeedbackType,
    ValidationStatus,
)


class ReviewItemResponse(BaseModel):
    """A single message in the review queue (negative feedback or flagged)."""

    message_id: str = Field(..., description="Message UUID")
    conversation_id: str = Field(..., description="Parent conversation UUID")
    user_email: str = Field("", description="Email of the user who sent the message")
    user_question: str = Field("", description="The preceding user question")
    ai_answer: str = Field(..., description="The AI-generated answer")
    sources_json: list[dict[str, object]] = Field(default_factory=list, description="Source citations")
    feedback: FeedbackType = Field(..., description="Feedback type")
    validation_status: ValidationStatus = Field(ValidationStatus.NONE, description="Output validation result")
    moderation_reason: str = Field("", description="Content moderation reason")
    moderation_matched_term: str = Field("", description="Specific term that triggered moderation")
    reviewed_at: datetime | None = Field(None, description="When the item was reviewed")
    reviewed_by: str = Field("", description="Reviewer user ID")
    created_at: datetime | None = Field(None, description="Message creation timestamp")


class NegativeFeedbackListResponse(BaseModel):
    """Paginated list of negatively-rated messages."""

    items: list[ReviewItemResponse] = Field(..., description="Review items")
    total: int = Field(..., description="Total matching count")
    limit: int = Field(..., description="Page size")
    offset: int = Field(..., description="Page offset")


class EscalationItemResponse(BaseModel):
    """A single escalated conversation in the review queue."""

    conversation_id: str = Field(..., description="Conversation UUID")
    user_email: str = Field("", description="Email of the user who started the conversation")
    trigger: EscalationTrigger = Field(..., description="Escalation trigger type")
    first_message: str = Field("", description="First user message (preview)")
    status: ConversationStatus = Field(..., description="Conversation status")
    started_at: datetime | None = Field(None, description="Conversation start time")


class EscalationListResponse(BaseModel):
    """Paginated list of escalated conversations."""

    items: list[EscalationItemResponse] = Field(..., description="Escalation items")
    total: int = Field(..., description="Total matching count")
    limit: int = Field(..., description="Page size")
    offset: int = Field(..., description="Page offset")


class FlaggedMessageListResponse(BaseModel):
    """Paginated list of flagged messages (output validation failures)."""

    items: list[ReviewItemResponse] = Field(..., description="Flagged items")
    total: int = Field(..., description="Total matching count")
    limit: int = Field(..., description="Page size")
    offset: int = Field(..., description="Page offset")


class ReviewStatsResponse(BaseModel):
    """Aggregate counts for the review queue badge."""

    unreviewed_negative: int = Field(..., description="Unreviewed negative feedback count")
    unreviewed_flagged: int = Field(..., description="Unreviewed flagged messages count")
    open_escalations: int = Field(..., description="Unresolved escalated conversations count")
    unresolved_failed_queries: int = Field(0, description="Unresolved failed queries count")


class ReviewActionResponse(BaseModel):
    """Response after marking a message as reviewed."""

    message_id: str = Field(..., description="Reviewed message UUID")
    reviewed_at: datetime = Field(..., description="Review timestamp")
    reviewed_by: str = Field(..., description="Reviewer user ID")
