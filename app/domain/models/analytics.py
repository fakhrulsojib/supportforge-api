"""Domain models for analytics.

Pure Pydantic models for analytics data — NO framework imports.
Used by AnalyticsService and the analytics API endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DailyStatEntry(BaseModel):
    """A single day's conversation and message statistics."""

    date: str = Field(..., description="ISO date string (YYYY-MM-DD)")
    total_conversations: int = Field(0, ge=0, description="Conversations started on this day")
    total_messages: int = Field(0, ge=0, description="Messages sent on this day")


class IntentEntry(BaseModel):
    """A single topic/intent with its occurrence count."""

    name: str = Field(..., min_length=1, description="Topic or document name")
    count: int = Field(0, ge=0, description="Number of occurrences")


class SatisfactionSummary(BaseModel):
    """Aggregated feedback satisfaction metrics."""

    positive: int = Field(0, ge=0, description="Positive feedback count")
    negative: int = Field(0, ge=0, description="Negative feedback count")
    total: int = Field(0, ge=0, description="Total feedback count (positive + negative)")
    rate: float = Field(0.0, ge=0.0, le=1.0, description="Satisfaction rate (positive / total)")
