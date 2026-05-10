"""Failed Query API schemas — request/response DTOs.

Admin-only endpoints for reviewing and resolving failed queries.
Used by ``app.api.v1.failed_queries`` router.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003 — Pydantic needs runtime access

from pydantic import BaseModel, Field

from app.domain.models.enums import (  # noqa: TCH001 — Pydantic runtime
    EscalationTrigger,
    FailureReason,
)


class FailedQueryResponse(BaseModel):
    """A single failed query in the admin view."""

    id: str = Field(..., description="Failed query UUID")
    tenant_id: str = Field(..., description="Tenant UUID")
    conversation_id: str = Field(..., description="Parent conversation UUID")
    message_id: str = Field("", description="Associated message UUID")
    query_text: str = Field(..., description="The user's original question")
    failure_reason: FailureReason = Field(..., description="Why the query failed")
    retrieved_doc_count: int = Field(0, description="Number of docs retrieved")
    max_relevance_score: float = Field(0.0, description="Highest relevance score")
    escalation_trigger: EscalationTrigger = Field(
        EscalationTrigger.NONE, description="What triggered escalation",
    )
    created_at: datetime | None = Field(None, description="When the query failed")
    resolved_at: datetime | None = Field(None, description="When the query was resolved")
    resolved_by: str = Field("", description="Admin user ID who resolved it")


class FailedQueryListResponse(BaseModel):
    """Paginated list of failed queries."""

    items: list[FailedQueryResponse] = Field(..., description="Failed query items")
    total: int = Field(..., description="Total matching count")
    limit: int = Field(..., description="Page size")
    offset: int = Field(..., description="Page offset")


class FailedQueryResolveResponse(BaseModel):
    """Response after resolving a failed query."""

    id: str = Field(..., description="Resolved failed query UUID")
    resolved_at: datetime = Field(..., description="Resolution timestamp")
    resolved_by: str = Field(..., description="Resolver user ID")


class FailedQueryStatsResponse(BaseModel):
    """Aggregated failed query statistics."""

    total_unresolved: int = Field(..., description="Unresolved failed queries count")
    reason_breakdown: dict[str, int] = Field(
        default_factory=dict, description="Count per failure reason",
    )
    top_queries: list[dict[str, object]] = Field(
        default_factory=list, description="Top 10 repeated failed queries",
    )
    daily_trend: list[dict[str, object]] = Field(
        default_factory=list, description="Daily failed query counts (last 30d)",
    )
