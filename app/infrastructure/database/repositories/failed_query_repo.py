"""SQLAlchemy implementation of FailedQueryRepository."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import cast, func, select
from sqlalchemy.types import Date

from app.domain.interfaces.repository import FailedQueryRepository
from app.domain.models.enums import FailureReason
from app.domain.models.failed_query import FailedQuery
from app.infrastructure.database.models import FailedQueryModel

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class SQLFailedQueryRepository(FailedQueryRepository):
    """Concrete failed-query repository backed by PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def _to_domain(self, model: FailedQueryModel) -> FailedQuery:
        """Convert ORM model to domain model."""
        return FailedQuery(
            id=model.id,
            tenant_id=model.tenant_id,
            conversation_id=model.conversation_id,
            message_id=model.message_id,
            query_text=model.query_text,
            failure_reason=model.failure_reason,
            retrieved_doc_count=model.retrieved_doc_count,
            max_relevance_score=model.max_relevance_score,
            escalation_trigger=model.escalation_trigger,
            created_at=model.created_at,
            resolved_at=model.resolved_at,
            resolved_by=model.resolved_by,
        )

    async def create(self, failed_query: FailedQuery) -> FailedQuery:
        """Persist a new failed query record.

        Args:
            failed_query: Domain model with query details.

        Returns:
            The persisted FailedQuery with generated ID.
        """
        model = FailedQueryModel(
            tenant_id=failed_query.tenant_id,
            conversation_id=failed_query.conversation_id,
            message_id=failed_query.message_id,
            query_text=failed_query.query_text,
            failure_reason=failed_query.failure_reason,
            retrieved_doc_count=failed_query.retrieved_doc_count,
            max_relevance_score=failed_query.max_relevance_score,
            escalation_trigger=failed_query.escalation_trigger,
        )
        self._session.add(model)
        await self._session.flush()
        return self._to_domain(model)

    async def get_by_id(self, query_id: str) -> FailedQuery | None:
        """Get a failed query by ID."""
        result = await self._session.get(FailedQueryModel, query_id)
        return self._to_domain(result) if result else None

    async def list_by_tenant(
        self,
        tenant_id: str,
        *,
        failure_reason: FailureReason | None = None,
        resolved: bool | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[FailedQuery], int]:
        """List failed queries for a tenant with filters.

        Args:
            tenant_id: Tenant context for isolation.
            failure_reason: Optional filter by failure reason.
            resolved: If True only resolved, if False only unresolved, None = all.
            start_date: Optional ISO date lower bound.
            end_date: Optional ISO date upper bound.
            limit: Page size.
            offset: Page offset.

        Returns:
            Tuple of (failed_queries, total_count).
        """
        base = select(FailedQueryModel).where(
            FailedQueryModel.tenant_id == tenant_id,
        )

        if failure_reason is not None:
            base = base.where(FailedQueryModel.failure_reason == failure_reason)

        if resolved is True:
            base = base.where(FailedQueryModel.resolved_at.isnot(None))
        elif resolved is False:
            base = base.where(FailedQueryModel.resolved_at.is_(None))

        if start_date:
            base = base.where(FailedQueryModel.created_at >= start_date)
        if end_date:
            base = base.where(FailedQueryModel.created_at <= end_date)

        # Count
        count_stmt = select(func.count()).select_from(base.subquery())
        count_result = await self._session.execute(count_stmt)
        total = count_result.scalar() or 0

        # Paginated results
        stmt = base.order_by(FailedQueryModel.created_at.desc()).limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return [self._to_domain(m) for m in result.scalars().all()], total

    async def mark_resolved(
        self, query_id: str, resolved_by: str,
    ) -> FailedQuery | None:
        """Mark a failed query as resolved.

        Args:
            query_id: Failed query UUID.
            resolved_by: Admin user ID who resolved it.

        Returns:
            Updated FailedQuery, or None if not found.
        """
        model = await self._session.get(FailedQueryModel, query_id)
        if not model:
            return None
        model.resolved_at = datetime.now(timezone.utc)
        model.resolved_by = resolved_by
        await self._session.flush()
        return self._to_domain(model)

    async def count_unresolved(self, tenant_id: str) -> int:
        """Count unresolved failed queries for a tenant."""
        stmt = (
            select(func.count())
            .select_from(FailedQueryModel)
            .where(
                FailedQueryModel.tenant_id == tenant_id,
                FailedQueryModel.resolved_at.is_(None),
            )
        )
        result = await self._session.execute(stmt)
        return result.scalar() or 0

    async def get_stats(self, tenant_id: str) -> dict[str, Any]:
        """Get aggregated failed query statistics for a tenant.

        Returns a dict with:
        - reason_breakdown: {failure_reason: count}
        - top_queries: [{query_text, count}] (top 10 repeated)
        - daily_trend: [{date, count}] (last 30 days)

        All aggregation uses SQL — NOT Python-side aggregation.
        """
        # 1. Reason breakdown (unresolved only)
        reason_stmt = (
            select(
                FailedQueryModel.failure_reason,
                func.count().label("count"),
            )
            .where(
                FailedQueryModel.tenant_id == tenant_id,
                FailedQueryModel.resolved_at.is_(None),
            )
            .group_by(FailedQueryModel.failure_reason)
        )
        reason_result = await self._session.execute(reason_stmt)
        reason_breakdown: dict[str, int] = {}
        for row in reason_result.all():
            reason_val = row[0].value if hasattr(row[0], "value") else str(row[0])
            reason_breakdown[reason_val] = row[1]

        # 2. Top 10 repeated failed queries (unresolved)
        top_stmt = (
            select(
                FailedQueryModel.query_text,
                func.count().label("count"),
            )
            .where(
                FailedQueryModel.tenant_id == tenant_id,
                FailedQueryModel.resolved_at.is_(None),
            )
            .group_by(FailedQueryModel.query_text)
            .order_by(func.count().desc())
            .limit(10)
        )
        top_result = await self._session.execute(top_stmt)
        top_queries = [
            {"query_text": row[0], "count": row[1]}
            for row in top_result.all()
        ]

        # 3. Daily trend (last 30 days, all queries)
        daily_stmt = (
            select(
                cast(FailedQueryModel.created_at, Date).label("date"),
                func.count().label("count"),
            )
            .where(FailedQueryModel.tenant_id == tenant_id)
            .group_by(cast(FailedQueryModel.created_at, Date))
            .order_by(cast(FailedQueryModel.created_at, Date).desc())
            .limit(30)
        )
        daily_result = await self._session.execute(daily_stmt)
        daily_trend = [
            {"date": str(row[0]), "count": row[1]}
            for row in daily_result.all()
        ]

        return {
            "reason_breakdown": reason_breakdown,
            "top_queries": top_queries,
            "daily_trend": daily_trend,
        }
