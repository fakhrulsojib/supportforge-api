"""Analytics domain service — orchestrates analytics data retrieval.

Validates input parameters and delegates to the AnalyticsRepository port.
Pure domain layer — NO framework imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from app.domain.interfaces.repository import AnalyticsRepository
    from app.domain.models.analytics import DailyStatEntry, IntentEntry, SatisfactionSummary

logger = structlog.get_logger(__name__)

# ── Parameter bounds ────────────────────────────────────────────
_MIN_DAYS = 1
_MAX_DAYS = 365
_DEFAULT_DAYS = 30
_MIN_LIMIT = 1
_MAX_LIMIT = 100
_DEFAULT_LIMIT = 10


class AnalyticsService:
    """Domain service for analytics dashboard data.

    Attributes:
        _analytics_repo: Repository for analytics data aggregation.
    """

    def __init__(self, analytics_repo: AnalyticsRepository) -> None:
        self._analytics_repo = analytics_repo

    async def get_daily_stats(
        self, tenant_id: str, *, days: int = _DEFAULT_DAYS,
    ) -> list[DailyStatEntry]:
        """Retrieve daily conversation and message statistics.

        Args:
            tenant_id: Tenant context for isolation.
            days: Number of days to look back (clamped to 1–365).

        Returns:
            List of DailyStatEntry ordered by date ascending.
        """
        # Clamp days to valid range
        days = max(_MIN_DAYS, min(_MAX_DAYS, days))

        logger.info(
            "analytics_daily_stats_requested",
            tenant_id=tenant_id,
            days=days,
        )

        return await self._analytics_repo.get_daily_stats(tenant_id, days=days)

    async def get_top_intents(
        self, tenant_id: str, *, limit: int = _DEFAULT_LIMIT,
    ) -> list[IntentEntry]:
        """Retrieve top topics by occurrence frequency.

        Args:
            tenant_id: Tenant context for isolation.
            limit: Maximum number of intents (clamped to 1–100).

        Returns:
            List of IntentEntry sorted by count descending.
        """
        # Clamp limit to valid range
        limit = max(_MIN_LIMIT, min(_MAX_LIMIT, limit))

        logger.info(
            "analytics_top_intents_requested",
            tenant_id=tenant_id,
            limit=limit,
        )

        return await self._analytics_repo.get_top_intents(tenant_id, limit=limit)

    async def get_satisfaction_summary(
        self, tenant_id: str,
    ) -> SatisfactionSummary:
        """Retrieve aggregated satisfaction metrics.

        Args:
            tenant_id: Tenant context for isolation.

        Returns:
            SatisfactionSummary with positive, negative, total, rate.
        """
        logger.info(
            "analytics_satisfaction_requested",
            tenant_id=tenant_id,
        )

        return await self._analytics_repo.get_satisfaction_summary(tenant_id)
