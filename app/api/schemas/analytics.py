"""Analytics API schemas — request/response DTOs.

Admin-only endpoints for viewing analytics dashboard data.
Used by ``app.api.v1.analytics`` router.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DailyStatEntrySchema(BaseModel):
    """A single day's statistics for the conversation chart."""

    date: str = Field(..., description="ISO date string (YYYY-MM-DD)")
    total_conversations: int = Field(0, description="Conversations started this day")
    total_messages: int = Field(0, description="Messages sent this day")


class DailyStatsResponse(BaseModel):
    """Response for GET /api/v1/analytics/daily-stats."""

    stats: list[DailyStatEntrySchema] = Field(
        default_factory=list, description="Daily statistics entries",
    )


class IntentEntrySchema(BaseModel):
    """A single intent/topic entry for the topic cloud."""

    name: str = Field(..., description="Topic or document name")
    count: int = Field(0, description="Number of occurrences")


class TopIntentsResponse(BaseModel):
    """Response for GET /api/v1/analytics/top-intents."""

    intents: list[IntentEntrySchema] = Field(
        default_factory=list, description="Top intent entries",
    )


class SatisfactionResponse(BaseModel):
    """Response for GET /api/v1/analytics/satisfaction."""

    positive: int = Field(0, description="Positive feedback count")
    negative: int = Field(0, description="Negative feedback count")
    total: int = Field(0, description="Total feedback count")
    rate: float = Field(0.0, description="Satisfaction rate (0.0–1.0)")
