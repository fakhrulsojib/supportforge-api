"""Chat endpoint request/response schemas."""

from __future__ import annotations

from datetime import datetime, timezone  # noqa: TCH003 — Pydantic needs runtime access

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for POST /api/v1/chat."""

    message: str = Field(..., min_length=1, max_length=10000, description="User's message")
    conversation_id: str | None = Field(None, description="Existing conversation ID for context")


class SourceCitation(BaseModel):
    """A source citation for a generated answer."""

    content: str = Field(..., description="Relevant excerpt from the source document")
    score: float = Field(0.0, description="Relevance score (0.0 - 1.0)")
    id: str = Field("", description="Source document ID")


class ChatResponse(BaseModel):
    """Response body for POST /api/v1/chat."""

    answer: str = Field(..., description="AI-generated response")
    conversation_id: str = Field(..., description="Conversation ID for continuity")
    sources: list[SourceCitation] = Field(default_factory=list, description="Source citations")
    escalated: bool = Field(False, description="Whether the query was escalated to a human")
    escalation_reason: str = Field("", description="Reason for escalation, if any")
    model_used: str = Field("", description="LLM model used for generation")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Response timestamp")
