"""Unit tests for review queue API schemas."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.api.schemas.review import (
    EscalationItemResponse,
    EscalationListResponse,
    FlaggedMessageListResponse,
    NegativeFeedbackListResponse,
    ReviewActionResponse,
    ReviewItemResponse,
    ReviewStatsResponse,
)
from app.domain.models.enums import (
    ConversationStatus,
    EscalationTrigger,
    FeedbackType,
    ValidationStatus,
)


class TestReviewItemResponse:
    """Tests for ReviewItemResponse schema."""

    def test_valid_review_item(self) -> None:
        """Should accept all required fields."""
        item = ReviewItemResponse(
            message_id="msg-1",
            conversation_id="conv-1",
            user_question="How do I reset my password?",
            ai_answer="You can reset your password by clicking...",
            feedback=FeedbackType.NEGATIVE,
            validation_status=ValidationStatus.NONE,
            created_at=datetime.now(timezone.utc),
        )
        assert item.message_id == "msg-1"
        assert item.feedback == FeedbackType.NEGATIVE
        assert item.reviewed_at is None
        assert item.reviewed_by == ""

    def test_review_item_with_review_info(self) -> None:
        """Should accept reviewed_at and reviewed_by."""
        now = datetime.now(timezone.utc)
        item = ReviewItemResponse(
            message_id="msg-1",
            conversation_id="conv-1",
            ai_answer="Answer",
            feedback=FeedbackType.NEGATIVE,
            reviewed_at=now,
            reviewed_by="admin-1",
        )
        assert item.reviewed_at == now
        assert item.reviewed_by == "admin-1"

    def test_review_item_missing_required(self) -> None:
        """Should reject missing required fields."""
        with pytest.raises(ValidationError):
            ReviewItemResponse()  # type: ignore[call-arg]

    def test_review_item_default_sources(self) -> None:
        """Should default sources_json to empty list."""
        item = ReviewItemResponse(
            message_id="msg-1",
            conversation_id="conv-1",
            ai_answer="Answer",
            feedback=FeedbackType.NEGATIVE,
        )
        assert item.sources_json == []


class TestNegativeFeedbackListResponse:
    """Tests for NegativeFeedbackListResponse schema."""

    def test_valid_list(self) -> None:
        """Should accept items with pagination metadata."""
        item = ReviewItemResponse(
            message_id="msg-1",
            conversation_id="conv-1",
            ai_answer="Answer",
            feedback=FeedbackType.NEGATIVE,
        )
        resp = NegativeFeedbackListResponse(
            items=[item], total=1, limit=50, offset=0,
        )
        assert len(resp.items) == 1
        assert resp.total == 1

    def test_empty_list(self) -> None:
        """Should accept empty items list."""
        resp = NegativeFeedbackListResponse(
            items=[], total=0, limit=50, offset=0,
        )
        assert resp.items == []
        assert resp.total == 0


class TestEscalationItemResponse:
    """Tests for EscalationItemResponse schema."""

    def test_valid_escalation(self) -> None:
        """Should accept all escalation fields."""
        item = EscalationItemResponse(
            conversation_id="conv-1",
            trigger=EscalationTrigger.SENTIMENT,
            first_message="This is frustrating!",
            status=ConversationStatus.ESCALATED,
            started_at=datetime.now(timezone.utc),
        )
        assert item.trigger == EscalationTrigger.SENTIMENT
        assert item.status == ConversationStatus.ESCALATED

    def test_escalation_missing_required(self) -> None:
        """Should reject missing required fields."""
        with pytest.raises(ValidationError):
            EscalationItemResponse()  # type: ignore[call-arg]


class TestEscalationListResponse:
    """Tests for EscalationListResponse schema."""

    def test_valid_list(self) -> None:
        """Should accept items with pagination."""
        item = EscalationItemResponse(
            conversation_id="conv-1",
            trigger=EscalationTrigger.EXPLICIT_REQUEST,
            status=ConversationStatus.ESCALATED,
        )
        resp = EscalationListResponse(
            items=[item], total=1, limit=50, offset=0,
        )
        assert len(resp.items) == 1


class TestFlaggedMessageListResponse:
    """Tests for FlaggedMessageListResponse schema."""

    def test_valid_list(self) -> None:
        """Should accept flagged items with pagination."""
        item = ReviewItemResponse(
            message_id="msg-1",
            conversation_id="conv-1",
            ai_answer="Flagged answer",
            feedback=FeedbackType.NONE,
            validation_status=ValidationStatus.FLAGGED,
        )
        resp = FlaggedMessageListResponse(
            items=[item], total=1, limit=50, offset=0,
        )
        assert resp.items[0].validation_status == ValidationStatus.FLAGGED


class TestReviewStatsResponse:
    """Tests for ReviewStatsResponse schema."""

    def test_valid_stats(self) -> None:
        """Should accept all count fields."""
        stats = ReviewStatsResponse(
            unreviewed_negative=5,
            unreviewed_flagged=2,
            open_escalations=3,
        )
        assert stats.unreviewed_negative == 5
        assert stats.unreviewed_flagged == 2
        assert stats.open_escalations == 3

    def test_stats_missing_required(self) -> None:
        """Should reject missing fields."""
        with pytest.raises(ValidationError):
            ReviewStatsResponse()  # type: ignore[call-arg]


class TestReviewActionResponse:
    """Tests for ReviewActionResponse schema."""

    def test_valid_action(self) -> None:
        """Should accept mark-reviewed response."""
        now = datetime.now(timezone.utc)
        action = ReviewActionResponse(
            message_id="msg-1",
            reviewed_at=now,
            reviewed_by="admin-1",
        )
        assert action.message_id == "msg-1"
        assert action.reviewed_at == now
        assert action.reviewed_by == "admin-1"

    def test_action_missing_required(self) -> None:
        """Should reject missing fields."""
        with pytest.raises(ValidationError):
            ReviewActionResponse()  # type: ignore[call-arg]
