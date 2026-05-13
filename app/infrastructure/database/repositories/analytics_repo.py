"""SQLAlchemy implementation of AnalyticsRepository.

Provides real-time SQL aggregation queries for the analytics dashboard.
All queries are tenant-scoped via ``conversations.tenant_id`` since
messages do not have a direct ``tenant_id`` column.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import cast, func, select
from sqlalchemy.types import Date

from app.domain.interfaces.repository import AnalyticsRepository
from app.domain.models.analytics import DailyStatEntry, IntentEntry, SatisfactionSummary
from app.domain.models.enums import FeedbackType, MessageRole
from app.infrastructure.database.models import ConversationModel, MessageModel

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class SQLAnalyticsRepository(AnalyticsRepository):
    """Concrete analytics repository backed by PostgreSQL.

    All aggregation is performed via SQL queries on indexed columns.
    Message queries JOIN through ``conversations`` for tenant isolation.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_daily_stats(
        self, tenant_id: str, *, days: int = 30,
    ) -> list[DailyStatEntry]:
        """Aggregate daily conversation and message counts.

        Runs two grouped queries:
        1. Conversations per day (GROUP BY DATE(started_at))
        2. Messages per day (GROUP BY DATE(messages.created_at) via JOIN)

        Missing days in the range are filled with zero counts.

        Args:
            tenant_id: Tenant context for isolation.
            days: Number of days to look back from today.

        Returns:
            List of DailyStatEntry ordered by date ascending.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        # 1. Conversation counts per day
        conv_stmt = (
            select(
                cast(ConversationModel.started_at, Date).label("date"),
                func.count().label("count"),
            )
            .where(
                ConversationModel.tenant_id == tenant_id,
                ConversationModel.started_at >= cutoff,
            )
            .group_by(cast(ConversationModel.started_at, Date))
        )
        conv_result = await self._session.execute(conv_stmt)
        conv_by_date: dict[str, int] = {
            str(row.date): row.count for row in conv_result.all()
        }

        # 2. Message counts per day (JOIN through conversations for tenant scope)
        msg_stmt = (
            select(
                cast(MessageModel.created_at, Date).label("date"),
                func.count().label("count"),
            )
            .join(
                ConversationModel,
                MessageModel.conversation_id == ConversationModel.id,
            )
            .where(
                ConversationModel.tenant_id == tenant_id,
                MessageModel.created_at >= cutoff,
            )
            .group_by(cast(MessageModel.created_at, Date))
        )
        msg_result = await self._session.execute(msg_stmt)
        msg_by_date: dict[str, int] = {
            str(row.date): row.count for row in msg_result.all()
        }

        # 3. Merge and fill missing days
        all_dates: set[str] = set(conv_by_date.keys()) | set(msg_by_date.keys())

        # Also add any missing days in the range for a continuous chart
        today = datetime.now(timezone.utc).date()
        for i in range(days):
            day_str = str(today - timedelta(days=i))
            all_dates.add(day_str)

        stats: list[DailyStatEntry] = []
        for date_str in sorted(all_dates):
            stats.append(
                DailyStatEntry(
                    date=date_str,
                    total_conversations=conv_by_date.get(date_str, 0),
                    total_messages=msg_by_date.get(date_str, 0),
                )
            )

        return stats

    async def get_top_intents(
        self, tenant_id: str, *, limit: int = 10,
    ) -> list[IntentEntry]:
        """Extract top topics from assistant message sources.

        Reads ``sources_json`` from assistant messages, extracts the
        ``filename`` field from each source object, and aggregates by
        frequency. This acts as a proxy for "top intents" since the
        system doesn't have a dedicated intent classifier.

        Args:
            tenant_id: Tenant context for isolation.
            limit: Maximum number of intents to return.

        Returns:
            List of IntentEntry sorted by count descending.
        """
        # Select sources_json from assistant messages via JOIN
        stmt = (
            select(MessageModel.sources_json)
            .join(
                ConversationModel,
                MessageModel.conversation_id == ConversationModel.id,
            )
            .where(
                ConversationModel.tenant_id == tenant_id,
                MessageModel.role == MessageRole.ASSISTANT,
            )
        )
        result = await self._session.execute(stmt)

        # Aggregate filenames in Python (JSON arrays can't be easily
        # unnested in portable SQL across all PostgreSQL versions)
        filename_counter: Counter[str] = Counter()

        for (sources_raw,) in result.all():
            sources = sources_raw or []

            # Handle string-encoded JSON (shouldn't happen, but defensive)
            if isinstance(sources, str):
                try:
                    sources = json.loads(sources)
                except (json.JSONDecodeError, TypeError):
                    continue

            if not isinstance(sources, list):
                continue

            for source in sources:
                if isinstance(source, dict):
                    filename = source.get("filename", "")
                    if filename and filename != "Unknown source":
                        filename_counter[filename] += 1

        # Return top N
        return [
            IntentEntry(name=name, count=count)
            for name, count in filename_counter.most_common(limit)
        ]

    async def get_satisfaction_summary(
        self, tenant_id: str,
    ) -> SatisfactionSummary:
        """Aggregate positive and negative feedback counts.

        Counts messages with positive/negative feedback via JOIN through
        ``conversations`` for tenant isolation. Computes the satisfaction
        rate as ``positive / total`` (0.0 if no feedback).

        Args:
            tenant_id: Tenant context for isolation.

        Returns:
            SatisfactionSummary with counts and rate.
        """
        # Count positive feedback
        pos_stmt = (
            select(func.count())
            .select_from(MessageModel)
            .join(
                ConversationModel,
                MessageModel.conversation_id == ConversationModel.id,
            )
            .where(
                ConversationModel.tenant_id == tenant_id,
                MessageModel.feedback == FeedbackType.POSITIVE,
            )
        )
        pos_result = await self._session.execute(pos_stmt)
        positive = pos_result.scalar() or 0

        # Count negative feedback
        neg_stmt = (
            select(func.count())
            .select_from(MessageModel)
            .join(
                ConversationModel,
                MessageModel.conversation_id == ConversationModel.id,
            )
            .where(
                ConversationModel.tenant_id == tenant_id,
                MessageModel.feedback == FeedbackType.NEGATIVE,
            )
        )
        neg_result = await self._session.execute(neg_stmt)
        negative = neg_result.scalar() or 0

        total = positive + negative
        rate = positive / total if total > 0 else 0.0

        return SatisfactionSummary(
            positive=positive,
            negative=negative,
            total=total,
            rate=round(rate, 4),
        )
