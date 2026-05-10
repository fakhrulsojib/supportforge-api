"""Integration tests for review queue admin endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.security import create_access_token
from app.domain.models.conversation import Conversation, Message
from app.domain.models.enums import (
    ConversationStatus,
    EscalationTrigger,
    FeedbackType,
    MessageRole,
    UserRole,
    ValidationStatus,
)
from app.domain.models.user import User
from app.main import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

_JWT_SECRET = "change-me-to-another-random-secret"  # noqa: S105


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def admin_user() -> User:
    """Authenticated admin user."""
    return User(id="admin-1", tenant_id="t-1", email="admin@test.com", role=UserRole.ADMIN)


@pytest.fixture
def admin_token() -> str:
    """JWT for admin user."""
    return create_access_token(
        user_id="admin-1",
        tenant_id="t-1",
        role="admin",
        secret_key=_JWT_SECRET,
    )


@pytest.fixture
def viewer_user() -> User:
    """Authenticated viewer user (non-admin)."""
    return User(id="viewer-1", tenant_id="t-1", email="viewer@test.com", role=UserRole.VIEWER)


@pytest.fixture
def viewer_token() -> str:
    """JWT for viewer user."""
    return create_access_token(
        user_id="viewer-1",
        tenant_id="t-1",
        role="viewer",
        secret_key=_JWT_SECRET,
    )


@pytest.fixture
def mock_session() -> AsyncMock:
    """Mock DB session."""
    session = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def app_with_mocks(mock_session: AsyncMock) -> MagicMock:
    """App with mocked DB."""
    app = create_app()

    async def _gen() -> AsyncGenerator[AsyncMock, None]:
        yield mock_session

    from app.infrastructure.database.connection import get_async_session

    app.dependency_overrides[get_async_session] = _gen
    return app


@pytest.fixture
async def review_client(app_with_mocks: MagicMock) -> AsyncGenerator[AsyncClient, None]:
    """HTTP test client."""
    transport = ASGITransport(app=app_with_mocks)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest.fixture
def negative_message() -> Message:
    """Message with negative feedback."""
    return Message(
        id="msg-neg-1",
        conversation_id="conv-1",
        role=MessageRole.ASSISTANT,
        content="Here is a bad answer.",
        feedback=FeedbackType.NEGATIVE,
        validation_status=ValidationStatus.PASSED,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def user_question_message() -> Message:
    """User question message (precedes the negative message)."""
    return Message(
        id="msg-q-1",
        conversation_id="conv-1",
        role=MessageRole.USER,
        content="How do I reset my password?",
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def flagged_message() -> Message:
    """Message with flagged validation status."""
    return Message(
        id="msg-flag-1",
        conversation_id="conv-1",
        role=MessageRole.ASSISTANT,
        content="Flagged answer with hallucination.",
        feedback=FeedbackType.NONE,
        validation_status=ValidationStatus.FLAGGED,
        moderation_reason="hallucination_detected",
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def escalated_conversation() -> Conversation:
    """Escalated conversation."""
    return Conversation(
        id="conv-esc-1",
        tenant_id="t-1",
        user_id="user-1",
        status=ConversationStatus.ESCALATED,
        escalation_trigger=EscalationTrigger.SENTIMENT,
        started_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_conversation() -> Conversation:
    """Normal active conversation."""
    return Conversation(
        id="conv-1",
        tenant_id="t-1",
        user_id="user-1",
        status=ConversationStatus.ACTIVE,
    )


@pytest.fixture
def conv_owner_user() -> User:
    """The user who owns the sample conversation (for email resolution)."""
    return User(id="user-1", tenant_id="t-1", email="user@test.com", role=UserRole.VIEWER)


# ── RBAC Tests ───────────────────────────────────────────────────


class TestReviewRBAC:
    """Viewer/unauthenticated users cannot access review endpoints."""

    @pytest.mark.asyncio
    async def test_viewer_cannot_list_negative(
        self,
        review_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Viewer role should get 401 (insufficient permissions)."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await review_client.get(
                "/api/v1/admin/feedback/negative",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_viewer_cannot_list_escalations(
        self,
        review_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Viewer role should get 401 for escalations."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await review_client.get(
                "/api/v1/admin/escalations",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_viewer_cannot_list_flagged(
        self,
        review_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Viewer role should get 401 for flagged messages."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await review_client.get(
                "/api/v1/admin/flagged",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_viewer_cannot_mark_reviewed(
        self,
        review_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Viewer role should get 401 for mark-reviewed."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await review_client.patch(
                "/api/v1/admin/feedback/msg-1/review",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_viewer_cannot_get_stats(
        self,
        review_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Viewer role should get 401 for stats."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await review_client.get(
                "/api/v1/admin/feedback/stats",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(
        self,
        review_client: AsyncClient,
    ) -> None:
        """No token should get 401 (HTTPBearer rejects)."""
        response = await review_client.get("/api/v1/admin/feedback/negative")
        assert response.status_code == 401


# ── Negative Feedback Tests ──────────────────────────────────────


class TestListNegativeFeedback:
    """Tests for GET /api/v1/admin/feedback/negative."""

    @pytest.mark.asyncio
    async def test_list_negative_feedback_happy_path(
        self,
        review_client: AsyncClient,
        admin_token: str,
        admin_user: User,
        negative_message: Message,
        user_question_message: Message,
        sample_conversation: Conversation,
        conv_owner_user: User,
    ) -> None:
        """Admin should see negative feedback with user question context."""
        with (
            patch("app.api.v1.review.SQLMessageRepository") as mock_msg_cls,
            patch("app.api.v1.review.SQLConversationRepository") as mock_conv_cls,
            patch("app.api.v1.review.SQLUserRepository") as mock_review_user_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_msg = mock_msg_cls.return_value
            mock_msg.list_negative_feedback = AsyncMock(return_value=([negative_message], 1))
            mock_msg.get_preceding_user_messages_batch = AsyncMock(
                return_value={"msg-neg-1": user_question_message},
            )

            mock_conv = mock_conv_cls.return_value
            mock_conv.get_by_id = AsyncMock(return_value=sample_conversation)

            mock_review_user = mock_review_user_cls.return_value
            mock_review_user.get_by_id = AsyncMock(return_value=conv_owner_user)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await review_client.get(
                "/api/v1/admin/feedback/negative",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["message_id"] == "msg-neg-1"
        assert data["items"][0]["user_question"] == "How do I reset my password?"
        assert data["items"][0]["feedback"] == "negative"
        assert data["items"][0]["user_email"] == "user@test.com"

    @pytest.mark.asyncio
    async def test_list_negative_feedback_empty(
        self,
        review_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Should return empty list when no negative feedback exists."""
        with (
            patch("app.api.v1.review.SQLMessageRepository") as mock_msg_cls,
            patch("app.api.v1.review.SQLConversationRepository"),
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_msg = mock_msg_cls.return_value
            mock_msg.list_negative_feedback = AsyncMock(return_value=([], 0))
            mock_msg.get_preceding_user_messages_batch = AsyncMock(return_value={})

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await review_client.get(
                "/api/v1/admin/feedback/negative",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["items"] == []

    @pytest.mark.asyncio
    async def test_list_negative_with_reviewed_filter(
        self,
        review_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Should pass reviewed filter to repository."""
        with (
            patch("app.api.v1.review.SQLMessageRepository") as mock_msg_cls,
            patch("app.api.v1.review.SQLConversationRepository"),
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_msg = mock_msg_cls.return_value
            mock_msg.list_negative_feedback = AsyncMock(return_value=([], 0))
            mock_msg.get_preceding_user_messages_batch = AsyncMock(return_value={})

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await review_client.get(
                "/api/v1/admin/feedback/negative?reviewed=false",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        # Verify the reviewed filter was passed
        mock_msg.list_negative_feedback.assert_called_once()
        call_kwargs = mock_msg.list_negative_feedback.call_args
        assert call_kwargs.kwargs.get("reviewed") is False or call_kwargs[1].get("reviewed") is False


# ── Escalation Tests ─────────────────────────────────────────────


class TestListEscalations:
    """Tests for GET /api/v1/admin/escalations."""

    @pytest.mark.asyncio
    async def test_list_escalations_happy_path(
        self,
        review_client: AsyncClient,
        admin_token: str,
        admin_user: User,
        escalated_conversation: Conversation,
        user_question_message: Message,
        conv_owner_user: User,
    ) -> None:
        """Admin should see escalated conversations."""
        with (
            patch("app.api.v1.review.SQLConversationRepository") as mock_conv_cls,
            patch("app.api.v1.review.SQLMessageRepository") as mock_msg_cls,
            patch("app.api.v1.review.SQLUserRepository") as mock_review_user_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_conv = mock_conv_cls.return_value
            mock_conv.list_escalated = AsyncMock(return_value=([escalated_conversation], 1))

            mock_msg = mock_msg_cls.return_value
            mock_msg.list_by_conversation = AsyncMock(return_value=[user_question_message])

            mock_review_user = mock_review_user_cls.return_value
            mock_review_user.get_by_id = AsyncMock(return_value=conv_owner_user)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await review_client.get(
                "/api/v1/admin/escalations",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["conversation_id"] == "conv-esc-1"
        assert data["items"][0]["trigger"] == "sentiment"
        assert data["items"][0]["user_email"] == "user@test.com"

    @pytest.mark.asyncio
    async def test_list_escalations_with_trigger_filter(
        self,
        review_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Should pass trigger filter to repository."""
        with (
            patch("app.api.v1.review.SQLConversationRepository") as mock_conv_cls,
            patch("app.api.v1.review.SQLMessageRepository"),
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_conv = mock_conv_cls.return_value
            mock_conv.list_escalated = AsyncMock(return_value=([], 0))

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await review_client.get(
                "/api/v1/admin/escalations?trigger=sentiment",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200


# ── Flagged Messages Tests ───────────────────────────────────────


class TestListFlagged:
    """Tests for GET /api/v1/admin/flagged."""

    @pytest.mark.asyncio
    async def test_list_flagged_happy_path(
        self,
        review_client: AsyncClient,
        admin_token: str,
        admin_user: User,
        flagged_message: Message,
        user_question_message: Message,
        sample_conversation: Conversation,
        conv_owner_user: User,
    ) -> None:
        """Admin should see flagged messages."""
        with (
            patch("app.api.v1.review.SQLMessageRepository") as mock_msg_cls,
            patch("app.api.v1.review.SQLConversationRepository") as mock_conv_cls,
            patch("app.api.v1.review.SQLUserRepository") as mock_review_user_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_msg = mock_msg_cls.return_value
            mock_msg.list_flagged_messages = AsyncMock(return_value=([flagged_message], 1))
            mock_msg.get_preceding_user_messages_batch = AsyncMock(
                return_value={"msg-flag-1": user_question_message},
            )

            mock_conv = mock_conv_cls.return_value
            mock_conv.get_by_id = AsyncMock(return_value=sample_conversation)

            mock_review_user = mock_review_user_cls.return_value
            mock_review_user.get_by_id = AsyncMock(return_value=conv_owner_user)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await review_client.get(
                "/api/v1/admin/flagged",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["validation_status"] == "flagged"
        assert data["items"][0]["user_email"] == "user@test.com"


# ── Mark Reviewed Tests ──────────────────────────────────────────


class TestMarkReviewed:
    """Tests for PATCH /api/v1/admin/feedback/{message_id}/review."""

    @pytest.mark.asyncio
    async def test_mark_reviewed_happy_path(
        self,
        review_client: AsyncClient,
        admin_token: str,
        admin_user: User,
        negative_message: Message,
        sample_conversation: Conversation,
    ) -> None:
        """Should mark a message as reviewed."""
        now = datetime.now(timezone.utc)
        reviewed_msg = Message(
            id="msg-neg-1",
            conversation_id="conv-1",
            role=MessageRole.ASSISTANT,
            content="Bad answer",
            feedback=FeedbackType.NEGATIVE,
            reviewed_at=now,
            reviewed_by="admin-1",
        )

        with (
            patch("app.api.v1.review.SQLMessageRepository") as mock_msg_cls,
            patch("app.api.v1.review.SQLConversationRepository") as mock_conv_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_msg = mock_msg_cls.return_value
            mock_msg.get_by_id = AsyncMock(return_value=negative_message)
            mock_msg.update_review_status = AsyncMock(return_value=reviewed_msg)

            mock_conv = mock_conv_cls.return_value
            mock_conv.get_by_id = AsyncMock(return_value=sample_conversation)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await review_client.patch(
                "/api/v1/admin/feedback/msg-neg-1/review",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["message_id"] == "msg-neg-1"
        assert data["reviewed_by"] == "admin-1"
        assert data["reviewed_at"] is not None

    @pytest.mark.asyncio
    async def test_mark_reviewed_not_found(
        self,
        review_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Should return 404 for non-existent message."""
        with (
            patch("app.api.v1.review.SQLMessageRepository") as mock_msg_cls,
            patch("app.api.v1.review.SQLConversationRepository"),
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_msg = mock_msg_cls.return_value
            mock_msg.get_by_id = AsyncMock(return_value=None)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await review_client.patch(
                "/api/v1/admin/feedback/nonexistent/review",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_mark_reviewed_cross_tenant(
        self,
        review_client: AsyncClient,
        admin_token: str,
        admin_user: User,
        negative_message: Message,
    ) -> None:
        """Should return 404 for message from different tenant."""
        other_conv = Conversation(
            id="conv-1",
            tenant_id="other-tenant",
            user_id="u",
            status=ConversationStatus.ACTIVE,
        )

        with (
            patch("app.api.v1.review.SQLMessageRepository") as mock_msg_cls,
            patch("app.api.v1.review.SQLConversationRepository") as mock_conv_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_msg = mock_msg_cls.return_value
            mock_msg.get_by_id = AsyncMock(return_value=negative_message)

            mock_conv = mock_conv_cls.return_value
            mock_conv.get_by_id = AsyncMock(return_value=other_conv)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await review_client.patch(
                "/api/v1/admin/feedback/msg-neg-1/review",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_mark_reviewed_already_reviewed_overwrites(
        self,
        review_client: AsyncClient,
        admin_token: str,
        admin_user: User,
        sample_conversation: Conversation,
    ) -> None:
        """Re-reviewing an already reviewed message should update the timestamp."""
        original_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        new_time = datetime.now(timezone.utc)

        already_reviewed = Message(
            id="msg-neg-1",
            conversation_id="conv-1",
            role=MessageRole.ASSISTANT,
            content="Already reviewed answer",
            feedback=FeedbackType.NEGATIVE,
            reviewed_at=original_time,
            reviewed_by="other-admin",
        )

        re_reviewed = Message(
            id="msg-neg-1",
            conversation_id="conv-1",
            role=MessageRole.ASSISTANT,
            content="Already reviewed answer",
            feedback=FeedbackType.NEGATIVE,
            reviewed_at=new_time,
            reviewed_by="admin-1",
        )

        with (
            patch("app.api.v1.review.SQLMessageRepository") as mock_msg_cls,
            patch("app.api.v1.review.SQLConversationRepository") as mock_conv_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_msg = mock_msg_cls.return_value
            mock_msg.get_by_id = AsyncMock(return_value=already_reviewed)
            mock_msg.update_review_status = AsyncMock(return_value=re_reviewed)

            mock_conv = mock_conv_cls.return_value
            mock_conv.get_by_id = AsyncMock(return_value=sample_conversation)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await review_client.patch(
                "/api/v1/admin/feedback/msg-neg-1/review",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["reviewed_by"] == "admin-1"
        # Timestamp should be the new one, not the original
        assert data["reviewed_at"] != original_time.isoformat()


# ── Stats Tests ──────────────────────────────────────────────────


class TestReviewStats:
    """Tests for GET /api/v1/admin/feedback/stats."""

    @pytest.mark.asyncio
    async def test_stats_happy_path(
        self,
        review_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Should return aggregate review counts."""
        with (
            patch("app.api.v1.review.SQLMessageRepository") as mock_msg_cls,
            patch("app.api.v1.review.SQLConversationRepository") as mock_conv_cls,
            patch("app.infrastructure.database.repositories.failed_query_repo.SQLFailedQueryRepository") as mock_fq_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_msg = mock_msg_cls.return_value
            mock_msg.count_unreviewed_negative = AsyncMock(return_value=5)
            mock_msg.count_unreviewed_flagged = AsyncMock(return_value=2)

            mock_conv = mock_conv_cls.return_value
            mock_conv.count_open_escalations = AsyncMock(return_value=3)

            mock_fq = mock_fq_cls.return_value
            mock_fq.count_unresolved = AsyncMock(return_value=7)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await review_client.get(
                "/api/v1/admin/feedback/stats",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["unreviewed_negative"] == 5
        assert data["unreviewed_flagged"] == 2
        assert data["open_escalations"] == 3
        assert data["unresolved_failed_queries"] == 7

