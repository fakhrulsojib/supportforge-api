"""Unit tests for FailedQuery domain model and FailureReason enum."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.domain.models.enums import EscalationTrigger, FailureReason
from app.domain.models.failed_query import FailedQuery


class TestFailureReasonEnum:
    """Tests for the FailureReason enum."""

    def test_enum_values(self) -> None:
        """All expected failure reasons are present."""
        assert FailureReason.NO_DOCS.value == "no_docs"
        assert FailureReason.LOW_RELEVANCE.value == "low_relevance"
        assert FailureReason.LLM_ERROR.value == "llm_error"
        assert FailureReason.TIMEOUT.value == "timeout"

    def test_enum_count(self) -> None:
        """Exactly 4 failure reasons defined."""
        assert len(FailureReason) == 4

    def test_enum_is_str(self) -> None:
        """FailureReason values are strings (str, Enum)."""
        for member in FailureReason:
            assert isinstance(member.value, str)


class TestFailedQueryModel:
    """Tests for the FailedQuery domain model."""

    def test_create_with_required_fields(self) -> None:
        """Create a FailedQuery with minimum required fields."""
        fq = FailedQuery(
            query_text="Where is my order?",
            failure_reason=FailureReason.NO_DOCS,
        )
        assert fq.query_text == "Where is my order?"
        assert fq.failure_reason == FailureReason.NO_DOCS

    def test_defaults(self) -> None:
        """All optional fields have correct defaults."""
        fq = FailedQuery(
            query_text="test",
            failure_reason=FailureReason.NO_DOCS,
        )
        assert fq.id == ""
        assert fq.tenant_id == ""
        assert fq.conversation_id == ""
        assert fq.message_id == ""
        assert fq.retrieved_doc_count == 0
        assert fq.max_relevance_score == 0.0
        assert fq.escalation_trigger == EscalationTrigger.NONE
        assert fq.created_at is None
        assert fq.resolved_at is None
        assert fq.resolved_by == ""

    def test_create_with_all_fields(self) -> None:
        """Create a FailedQuery with all fields populated."""
        now = datetime.now(timezone.utc)
        fq = FailedQuery(
            id="fq-123",
            tenant_id="t-1",
            conversation_id="c-1",
            message_id="m-1",
            query_text="How do I reset my password?",
            failure_reason=FailureReason.LOW_RELEVANCE,
            retrieved_doc_count=3,
            max_relevance_score=0.25,
            escalation_trigger=EscalationTrigger.NO_CONTEXT,
            created_at=now,
            resolved_at=now,
            resolved_by="admin-1",
        )
        assert fq.id == "fq-123"
        assert fq.tenant_id == "t-1"
        assert fq.conversation_id == "c-1"
        assert fq.message_id == "m-1"
        assert fq.query_text == "How do I reset my password?"
        assert fq.failure_reason == FailureReason.LOW_RELEVANCE
        assert fq.retrieved_doc_count == 3
        assert fq.max_relevance_score == 0.25
        assert fq.escalation_trigger == EscalationTrigger.NO_CONTEXT
        assert fq.created_at == now
        assert fq.resolved_at == now
        assert fq.resolved_by == "admin-1"

    def test_empty_query_text_rejected(self) -> None:
        """query_text cannot be empty string."""
        with pytest.raises(Exception):  # noqa: B017
            FailedQuery(
                query_text="",
                failure_reason=FailureReason.NO_DOCS,
            )

    def test_all_failure_reasons_accepted(self) -> None:
        """Each FailureReason value is accepted by the model."""
        for reason in FailureReason:
            fq = FailedQuery(query_text="test query", failure_reason=reason)
            assert fq.failure_reason == reason

    def test_all_escalation_triggers_accepted(self) -> None:
        """Each EscalationTrigger value is accepted by the model."""
        for trigger in EscalationTrigger:
            fq = FailedQuery(
                query_text="test query",
                failure_reason=FailureReason.NO_DOCS,
                escalation_trigger=trigger,
            )
            assert fq.escalation_trigger == trigger

    def test_negative_doc_count_accepted(self) -> None:
        """Negative retrieved_doc_count is accepted (no business constraint)."""
        fq = FailedQuery(
            query_text="test",
            failure_reason=FailureReason.NO_DOCS,
            retrieved_doc_count=-1,
        )
        assert fq.retrieved_doc_count == -1

    def test_zero_max_relevance_score(self) -> None:
        """Zero max_relevance_score is valid."""
        fq = FailedQuery(
            query_text="test",
            failure_reason=FailureReason.NO_DOCS,
            max_relevance_score=0.0,
        )
        assert fq.max_relevance_score == 0.0
