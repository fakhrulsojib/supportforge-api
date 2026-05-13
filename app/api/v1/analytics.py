"""Analytics API router — admin-only endpoints for dashboard data.

Provides daily statistics, top intents, and satisfaction metrics.
All endpoints enforce admin-only access via ``require_role(UserRole.ADMIN)``.
Tenant isolation is enforced through the authenticated user's tenant_id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends, Query

from app.api.schemas.analytics import (
    DailyStatEntrySchema,
    DailyStatsResponse,
    IntentEntrySchema,
    SatisfactionResponse,
    TopIntentsResponse,
)
from app.core.dependencies import require_role
from app.domain.models.enums import UserRole
from app.domain.services.analytics_service import AnalyticsService
from app.infrastructure.database.connection import get_async_session
from app.infrastructure.database.repositories.analytics_repo import (
    SQLAnalyticsRepository,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.domain.models.user import User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/daily-stats", response_model=DailyStatsResponse)
async def get_daily_stats(
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> DailyStatsResponse:
    """Get daily conversation and message statistics.

    Returns an array of per-day entries with conversation and message counts
    for the specified lookback period. Used by the ConversationChart component.

    Args:
        days: Number of days to look back (1–365, default 30).
        session: Database session.
        user: Authenticated admin user.

    Returns:
        Daily statistics for the chart.
    """
    repo = SQLAnalyticsRepository(session)
    service = AnalyticsService(repo)

    entries = await service.get_daily_stats(user.tenant_id, days=days)

    return DailyStatsResponse(
        stats=[
            DailyStatEntrySchema(
                date=e.date,
                total_conversations=e.total_conversations,
                total_messages=e.total_messages,
            )
            for e in entries
        ],
    )


@router.get("/top-intents", response_model=TopIntentsResponse)
async def get_top_intents(
    limit: int = Query(10, ge=1, le=100, description="Maximum number of intents"),
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> TopIntentsResponse:
    """Get top conversation topics by frequency.

    Extracts topic names from assistant message source citations and
    aggregates by frequency. Used by the TopicCloud component.

    Args:
        limit: Maximum number of intents to return (1–100, default 10).
        session: Database session.
        user: Authenticated admin user.

    Returns:
        Top intents sorted by count descending.
    """
    repo = SQLAnalyticsRepository(session)
    service = AnalyticsService(repo)

    intents = await service.get_top_intents(user.tenant_id, limit=limit)

    return TopIntentsResponse(
        intents=[
            IntentEntrySchema(name=i.name, count=i.count)
            for i in intents
        ],
    )


@router.get("/satisfaction", response_model=SatisfactionResponse)
async def get_satisfaction_rate(
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> SatisfactionResponse:
    """Get aggregated customer satisfaction metrics.

    Counts positive and negative feedback across all messages in the
    tenant's conversations. Computes satisfaction rate as
    ``positive / total``. Used by the SatisfactionGauge component.

    Args:
        session: Database session.
        user: Authenticated admin user.

    Returns:
        Satisfaction counts and rate.
    """
    repo = SQLAnalyticsRepository(session)
    service = AnalyticsService(repo)

    summary = await service.get_satisfaction_summary(user.tenant_id)

    return SatisfactionResponse(
        positive=summary.positive,
        negative=summary.negative,
        total=summary.total,
        rate=summary.rate,
    )
