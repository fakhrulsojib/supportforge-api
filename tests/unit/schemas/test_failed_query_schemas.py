"""Unit tests for FailedQuery schemas."""

from __future__ import annotations

from datetime import datetime, timezone

from app.api.schemas.failed_query import (
    FailedQueryListResponse,
    FailedQueryResolveResponse,
    FailedQueryResponse,
    FailedQueryStatsResponse,
)
from app.domain.models.enums import EscalationTrigger, FailureReason


class TestFailedQueryResponse:
    """Tests for FailedQueryResponse schema."""

    def test_create_response(self) -> None:
        """Can create a response with all fields."""
        now = datetime.now(timezone.utc)
        resp = FailedQueryResponse(
            id="fq-1",
            tenant_id="t-1",
            conversation_id="c-1",
            message_id="m-1",
            query_text="Where is my order?",
            failure_reason=FailureReason.NO_DOCS,
            retrieved_doc_count=0,
            max_relevance_score=0.0,
            escalation_trigger=EscalationTrigger.NO_CONTEXT,
            created_at=now,
            resolved_at=None,
            resolved_by="",
        )
        assert resp.id == "fq-1"
        assert resp.failure_reason == FailureReason.NO_DOCS
        assert resp.resolved_at is None

    def test_defaults(self) -> None:
        """Optional fields have correct defaults."""
        resp = FailedQueryResponse(
            id="fq-1",
            tenant_id="t-1",
            conversation_id="c-1",
            message_id="",
            query_text="test",
            failure_reason=FailureReason.TIMEOUT,
            retrieved_doc_count=0,
            max_relevance_score=0.0,
            escalation_trigger=EscalationTrigger.NONE,
        )
        assert resp.resolved_at is None
        assert resp.resolved_by == ""
        assert resp.created_at is None


class TestFailedQueryListResponse:
    """Tests for FailedQueryListResponse schema."""

    def test_empty_list(self) -> None:
        """Can create response with empty items list."""
        resp = FailedQueryListResponse(items=[], total=0, limit=50, offset=0)
        assert resp.items == []
        assert resp.total == 0

    def test_with_items(self) -> None:
        """Can create response with items."""
        item = FailedQueryResponse(
            id="fq-1",
            tenant_id="t-1",
            conversation_id="c-1",
            message_id="",
            query_text="test",
            failure_reason=FailureReason.NO_DOCS,
            retrieved_doc_count=0,
            max_relevance_score=0.0,
            escalation_trigger=EscalationTrigger.NONE,
        )
        resp = FailedQueryListResponse(items=[item], total=1, limit=50, offset=0)
        assert len(resp.items) == 1
        assert resp.total == 1


class TestFailedQueryResolveResponse:
    """Tests for FailedQueryResolveResponse schema."""

    def test_create(self) -> None:
        """Can create resolve response."""
        now = datetime.now(timezone.utc)
        resp = FailedQueryResolveResponse(
            id="fq-1",
            resolved_at=now,
            resolved_by="admin-1",
        )
        assert resp.id == "fq-1"
        assert resp.resolved_at == now
        assert resp.resolved_by == "admin-1"


class TestFailedQueryStatsResponse:
    """Tests for FailedQueryStatsResponse schema."""

    def test_create(self) -> None:
        """Can create stats response."""
        resp = FailedQueryStatsResponse(
            total_unresolved=5,
            reason_breakdown={"no_docs": 3, "timeout": 2},
            top_queries=[{"query_text": "test", "count": 5}],
            daily_trend=[{"date": "2026-01-01", "count": 3}],
        )
        assert resp.total_unresolved == 5
        assert len(resp.reason_breakdown) == 2
        assert len(resp.top_queries) == 1

    def test_empty_stats(self) -> None:
        """Stats with no data."""
        resp = FailedQueryStatsResponse(
            total_unresolved=0,
            reason_breakdown={},
            top_queries=[],
            daily_trend=[],
        )
        assert resp.total_unresolved == 0
