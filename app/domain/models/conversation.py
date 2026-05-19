"""Domain model for conversations and messages.

Pure Pydantic models — NO framework imports.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.models.enums import (
    ConversationStatus,
    EscalationTrigger,
    FeedbackType,
    MessageChannel,
    MessageRole,
    ValidationStatus,
)


class Conversation(BaseModel):
    """A conversation between a user and the AI assistant."""

    id: str = ""
    tenant_id: str = ""
    user_id: str = ""
    started_at: datetime | None = None
    ended_at: datetime | None = None
    status: ConversationStatus = ConversationStatus.ACTIVE
    escalation_trigger: EscalationTrigger = EscalationTrigger.NONE
    reviewed_at: datetime | None = None
    reviewed_by: str = ""


class Message(BaseModel):
    """A single message within a conversation."""

    id: str = ""
    conversation_id: str = ""
    role: MessageRole
    content: str = Field(..., min_length=1)
    thinking: str = ""
    sources_json: list[dict[str, object]] = Field(default_factory=list)
    model_used: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    feedback: FeedbackType = FeedbackType.NONE
    validation_status: ValidationStatus = ValidationStatus.NONE
    moderation_reason: str = ""
    moderation_matched_term: str = ""
    reviewed_at: datetime | None = None
    reviewed_by: str = ""
    created_at: datetime | None = None
    channel: MessageChannel = MessageChannel.TEXT
