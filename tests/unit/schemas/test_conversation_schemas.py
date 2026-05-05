"""Schema validation tests for conversation DTOs.

Tests edge cases for Pydantic field constraints:
  - enum validation (MessageRole, FeedbackType, ConversationStatus)
  - default values
  - required vs optional fields
  - nested list/dict types
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas.conversation import (
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationSummaryResponse,
    FeedbackRequest,
    MessageResponse,
)
from app.domain.models.enums import ConversationStatus, FeedbackType, MessageRole


class TestMessageResponse:
    """Edge-case validation for MessageResponse."""

    def test_valid_message(self) -> None:
        msg = MessageResponse(
            id="m-1",
            role=MessageRole.ASSISTANT,
            content="Hello!",
        )
        assert msg.sources_json == []
        assert msg.model_used == ""
        assert msg.tokens_in == 0
        assert msg.tokens_out == 0
        assert msg.feedback == FeedbackType.NONE
        assert msg.created_at is None

    def test_missing_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="id"):
            MessageResponse(role=MessageRole.USER, content="Hi")  # type: ignore[call-arg]

    def test_missing_role_rejected(self) -> None:
        with pytest.raises(ValidationError, match="role"):
            MessageResponse(id="m-1", content="Hi")  # type: ignore[call-arg]

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(ValidationError, match="role"):
            MessageResponse(id="m-1", role="invalid", content="Hi")

    def test_invalid_feedback_rejected(self) -> None:
        with pytest.raises(ValidationError, match="feedback"):
            MessageResponse(
                id="m-1",
                role=MessageRole.USER,
                content="Hi",
                feedback="invalid",
            )

    def test_with_sources(self) -> None:
        msg = MessageResponse(
            id="m-1",
            role=MessageRole.ASSISTANT,
            content="Answer",
            sources_json=[{"doc": "faq.pdf", "chunk": 1}],
        )
        assert len(msg.sources_json) == 1

    def test_with_token_counts(self) -> None:
        msg = MessageResponse(
            id="m-1",
            role=MessageRole.ASSISTANT,
            content="Answer",
            model_used="qwen3:4b",
            tokens_in=50,
            tokens_out=120,
        )
        assert msg.tokens_in == 50
        assert msg.tokens_out == 120


class TestConversationDetailResponse:
    """Edge-case validation for ConversationDetailResponse."""

    def test_valid_detail(self) -> None:
        resp = ConversationDetailResponse(
            id="c-1",
            tenant_id="t-1",
            status=ConversationStatus.ACTIVE,
        )
        assert resp.messages == []
        assert resp.user_id == ""

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValidationError, match="status"):
            ConversationDetailResponse(
                id="c-1",
                tenant_id="t-1",
                status="invalid",
            )

    def test_with_messages(self) -> None:
        resp = ConversationDetailResponse(
            id="c-1",
            tenant_id="t-1",
            status=ConversationStatus.ACTIVE,
            messages=[
                MessageResponse(
                    id="m-1",
                    role=MessageRole.USER,
                    content="Hello",
                )
            ],
        )
        assert len(resp.messages) == 1


class TestConversationSummaryResponse:
    """Edge-case validation for ConversationSummaryResponse."""

    def test_valid_summary(self) -> None:
        resp = ConversationSummaryResponse(
            id="c-1",
            tenant_id="t-1",
            status=ConversationStatus.RESOLVED,
        )
        assert resp.user_id == ""
        assert resp.started_at is None

    def test_missing_status_rejected(self) -> None:
        with pytest.raises(ValidationError, match="status"):
            ConversationSummaryResponse(id="c-1", tenant_id="t-1")  # type: ignore[call-arg]


class TestConversationListResponse:
    """Edge-case validation for ConversationListResponse."""

    def test_valid_list(self) -> None:
        resp = ConversationListResponse(
            conversations=[],
            total=0,
            limit=20,
            offset=0,
        )
        assert resp.total == 0

    def test_missing_total_rejected(self) -> None:
        with pytest.raises(ValidationError, match="total"):
            ConversationListResponse(conversations=[], limit=20, offset=0)  # type: ignore[call-arg]


class TestFeedbackRequest:
    """Edge-case validation for FeedbackRequest."""

    def test_positive_feedback(self) -> None:
        req = FeedbackRequest(feedback=FeedbackType.POSITIVE)
        assert req.feedback == FeedbackType.POSITIVE

    def test_negative_feedback(self) -> None:
        req = FeedbackRequest(feedback=FeedbackType.NEGATIVE)
        assert req.feedback == FeedbackType.NEGATIVE

    def test_invalid_feedback_rejected(self) -> None:
        with pytest.raises(ValidationError, match="feedback"):
            FeedbackRequest(feedback="thumbs-up")  # type: ignore[arg-type]

    def test_missing_feedback_rejected(self) -> None:
        with pytest.raises(ValidationError, match="feedback"):
            FeedbackRequest()  # type: ignore[call-arg]
