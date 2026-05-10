"""Failed Queries API router — admin-only endpoints for knowledge gap analytics.

Provides listing, filtering, and resolution endpoints for failed queries
(RAG pipeline escalations). All endpoints enforce admin-only access via
``require_role(UserRole.ADMIN)``. Tenant isolation is enforced through
the authenticated user's tenant_id.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends, Query

from app.api.schemas.failed_query import (
    FailedQueryListResponse,
    FailedQueryResolveResponse,
    FailedQueryResponse,
    FailedQueryStatsResponse,
)
from app.core.dependencies import require_role
from app.core.exceptions import SupportForgeError
from app.domain.models.enums import FailureReason, UserRole
from app.infrastructure.database.connection import get_async_session
from app.infrastructure.database.repositories.failed_query_repo import (
    SQLFailedQueryRepository,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.domain.models.user import User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/admin/failed-queries", tags=["failed-queries"])


@router.get("", response_model=FailedQueryListResponse)
async def list_failed_queries(
    failure_reason: str | None = Query(None, description="Filter by failure reason"),
    resolved: bool | None = Query(None, description="Filter: True=resolved, False=unresolved, None=all"),
    start_date: str | None = Query(None, description="ISO date lower bound"),
    end_date: str | None = Query(None, description="ISO date upper bound"),
    limit: int = Query(50, ge=1, le=100, description="Page size"),
    offset: int = Query(0, ge=0, description="Page offset"),
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> FailedQueryListResponse:
    """List failed queries for the admin's tenant.

    Args:
        failure_reason: Optional failure reason filter.
        resolved: Optional resolved/unresolved filter.
        start_date: Optional ISO date lower bound.
        end_date: Optional ISO date upper bound.
        limit: Page size (max 100).
        offset: Page offset.
        session: Database session.
        user: Authenticated admin user.

    Returns:
        Paginated list of failed queries.
    """
    repo = SQLFailedQueryRepository(session)

    # Parse failure_reason filter
    reason_enum: FailureReason | None = None
    if failure_reason:
        with contextlib.suppress(ValueError):
            reason_enum = FailureReason(failure_reason)

    failed_queries, total = await repo.list_by_tenant(
        tenant_id=user.tenant_id,
        failure_reason=reason_enum,
        resolved=resolved,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )

    items = [
        FailedQueryResponse(
            id=fq.id,
            tenant_id=fq.tenant_id,
            conversation_id=fq.conversation_id,
            message_id=fq.message_id,
            query_text=fq.query_text,
            failure_reason=fq.failure_reason,
            retrieved_doc_count=fq.retrieved_doc_count,
            max_relevance_score=fq.max_relevance_score,
            escalation_trigger=fq.escalation_trigger,
            created_at=fq.created_at,
            resolved_at=fq.resolved_at,
            resolved_by=fq.resolved_by,
        )
        for fq in failed_queries
    ]

    return FailedQueryListResponse(
        items=items, total=total, limit=limit, offset=offset,
    )


@router.patch("/{query_id}/resolve", response_model=FailedQueryResolveResponse)
async def resolve_failed_query(
    query_id: str,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> FailedQueryResolveResponse:
    """Mark a failed query as resolved.

    Sets ``resolved_at`` to current UTC time and ``resolved_by`` to
    the admin's user ID. Verifies tenant ownership before updating.

    Args:
        query_id: Failed query UUID to resolve.
        session: Database session.
        user: Authenticated admin user.

    Returns:
        Resolution confirmation with timestamp.

    Raises:
        SupportForgeError: If failed query not found or belongs to different tenant.
    """
    repo = SQLFailedQueryRepository(session)

    # Verify exists and tenant ownership
    existing = await repo.get_by_id(query_id)
    if not existing or existing.tenant_id != user.tenant_id:
        raise SupportForgeError(
            message=f"Failed query '{query_id}' not found",
            status_code=404,
            error_code="FAILED_QUERY_NOT_FOUND",
        )

    # Mark as resolved
    updated = await repo.mark_resolved(query_id, user.id)
    if not updated:
        raise SupportForgeError(
            message=f"Failed query '{query_id}' not found",
            status_code=404,
            error_code="FAILED_QUERY_NOT_FOUND",
        )

    await session.commit()

    logger.info(
        "failed_query_resolved",
        query_id=query_id,
        resolved_by=user.id,
    )

    return FailedQueryResolveResponse(
        id=updated.id,
        resolved_at=updated.resolved_at,  # type: ignore[arg-type]
        resolved_by=updated.resolved_by,
    )


@router.get("/stats", response_model=FailedQueryStatsResponse)
async def get_failed_query_stats(
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> FailedQueryStatsResponse:
    """Get aggregated failed query statistics.

    Returns unresolved count, failure reason breakdown,
    top 10 repeated failed queries, and daily trend.

    Args:
        session: Database session.
        user: Authenticated admin user.

    Returns:
        Aggregated failed query statistics.
    """
    repo = SQLFailedQueryRepository(session)

    total_unresolved = await repo.count_unresolved(user.tenant_id)
    stats = await repo.get_stats(user.tenant_id)

    return FailedQueryStatsResponse(
        total_unresolved=total_unresolved,
        reason_breakdown=stats["reason_breakdown"],
        top_queries=stats["top_queries"],
        daily_trend=stats["daily_trend"],
    )
