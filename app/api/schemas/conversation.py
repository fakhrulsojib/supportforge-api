"""Conversation API schemas — request/response DTOs."""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003 — Pydantic needs runtime access

from pydantic import BaseModel, Field

from app.domain.models.enums import ConversationStatus, FeedbackType, MessageRole  # noqa: TCH001 — Pydantic runtime


class MessageResponse(BaseModel):
    """A single message within a conversation."""

    id: str = Field(..., description="Message UUID")
    role: MessageRole = Field(..., description="Message author role")
    content: str = Field(..., description="Message text content")
    sources_json: list[dict[str, object]] = Field(default_factory=list, description="Source citations")
    model_used: str = Field("", description="LLM model used (assistant only)")
    tokens_in: int = Field(0, description="Input tokens used")
    tokens_out: int = Field(0, description="Output tokens generated")
    feedback: FeedbackType = Field(FeedbackType.NONE, description="User feedback")
    created_at: datetime | None = Field(None, description="Message timestamp")


class ConversationDetailResponse(BaseModel):
    """Full conversation with messages."""

    id: str = Field(..., description="Conversation UUID")
    tenant_id: str = Field(..., description="Tenant UUID")
    user_id: str = Field("", description="User UUID")
    status: ConversationStatus = Field(..., description="Conversation status")
    messages: list[MessageResponse] = Field(default_factory=list, description="Conversation messages")
    started_at: datetime | None = Field(None, description="Conversation start time")


class ConversationSummaryResponse(BaseModel):
    """Conversation summary for list views."""

    id: str = Field(..., description="Conversation UUID")
    tenant_id: str = Field(..., description="Tenant UUID")
    user_id: str = Field("", description="User UUID")
    status: ConversationStatus = Field(..., description="Conversation status")
    started_at: datetime | None = Field(None, description="Conversation start time")


class ConversationListResponse(BaseModel):
    """Paginated list of conversations."""

    conversations: list[ConversationSummaryResponse] = Field(..., description="List of conversations")
    total: int = Field(..., description="Total count")
    limit: int = Field(..., description="Page size")
    offset: int = Field(..., description="Page offset")


class FeedbackRequest(BaseModel):
    """Request body for updating message feedback."""

    feedback: FeedbackType = Field(..., description="Feedback type: positive or negative")
