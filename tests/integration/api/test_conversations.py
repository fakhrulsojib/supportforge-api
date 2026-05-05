"""Integration tests for conversation endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.security import create_access_token
from app.domain.models.conversation import Conversation, Message
from app.domain.models.enums import ConversationStatus, FeedbackType, MessageRole, UserRole
from app.domain.models.user import User
from app.main import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

_JWT_SECRET = "change-me-to-another-random-secret"  # noqa: S105


@pytest.fixture
def viewer_user() -> User:
    """Authenticated viewer user."""
    return User(id="user-1", tenant_id="t-1", email="viewer@test.com", role=UserRole.VIEWER)


@pytest.fixture
def viewer_token() -> str:
    """JWT for viewer user."""
    return create_access_token(
        user_id="user-1",
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
async def conv_client(app_with_mocks: MagicMock) -> AsyncGenerator[AsyncClient, None]:
    """HTTP test client."""
    transport = ASGITransport(app=app_with_mocks)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest.fixture
def sample_conversation() -> Conversation:
    """Sample conversation."""
    return Conversation(
        id="conv-1",
        tenant_id="t-1",
        user_id="user-1",
        status=ConversationStatus.ACTIVE,
    )


@pytest.fixture
def sample_message() -> Message:
    """Sample message."""
    return Message(
        id="msg-1",
        conversation_id="conv-1",
        role=MessageRole.ASSISTANT,
        content="Hello!",
    )


class TestListConversations:
    """Tests for GET /api/v1/conversations/."""

    @pytest.mark.asyncio
    async def test_list_returns_conversations(
        self,
        conv_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
        sample_conversation: Conversation,
    ) -> None:
        """Should return paginated conversations for tenant."""
        with (
            patch("app.api.v1.conversations.SQLConversationRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.list_by_tenant = AsyncMock(return_value=[sample_conversation])

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await conv_client.get(
                "/api/v1/conversations/",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["conversations"][0]["id"] == "conv-1"


class TestGetConversation:
    """Tests for GET /api/v1/conversations/{id}."""

    @pytest.mark.asyncio
    async def test_get_with_messages(
        self,
        conv_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
        sample_conversation: Conversation,
        sample_message: Message,
    ) -> None:
        """Should return conversation with messages."""
        with (
            patch("app.api.v1.conversations.SQLConversationRepository") as mock_conv_cls,
            patch("app.api.v1.conversations.SQLMessageRepository") as mock_msg_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_conv = mock_conv_cls.return_value
            mock_conv.get_by_id = AsyncMock(return_value=sample_conversation)

            mock_msg = mock_msg_cls.return_value
            mock_msg.list_by_conversation = AsyncMock(return_value=[sample_message])

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await conv_client.get(
                "/api/v1/conversations/conv-1",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "conv-1"
        assert len(data["messages"]) == 1

    @pytest.mark.asyncio
    async def test_get_not_found(
        self,
        conv_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Should return 404 for non-existent conversation."""
        with (
            patch("app.api.v1.conversations.SQLConversationRepository") as mock_conv_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_conv = mock_conv_cls.return_value
            mock_conv.get_by_id = AsyncMock(return_value=None)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await conv_client.get(
                "/api/v1/conversations/nonexistent",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_cross_tenant_returns_404(
        self,
        conv_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Should return 404 for conversation from different tenant."""
        other_tenant_conv = Conversation(
            id="conv-other",
            tenant_id="other-tenant",
            user_id="u",
            status=ConversationStatus.ACTIVE,
        )

        with (
            patch("app.api.v1.conversations.SQLConversationRepository") as mock_conv_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_conv = mock_conv_cls.return_value
            mock_conv.get_by_id = AsyncMock(return_value=other_tenant_conv)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await conv_client.get(
                "/api/v1/conversations/conv-other",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 404


class TestFeedback:
    """Tests for PATCH /api/v1/conversations/messages/{id}/feedback."""

    @pytest.mark.asyncio
    async def test_update_feedback(
        self,
        conv_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Should update message feedback."""
        updated_msg = Message(
            id="msg-1",
            conversation_id="conv-1",
            role=MessageRole.ASSISTANT,
            content="Hello!",
            feedback=FeedbackType.POSITIVE,
        )

        with (
            patch("app.api.v1.conversations.SQLMessageRepository") as mock_msg_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_msg = mock_msg_cls.return_value
            mock_msg.update_feedback = AsyncMock(return_value=updated_msg)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await conv_client.patch(
                "/api/v1/conversations/messages/msg-1/feedback",
                json={"feedback": "positive"},
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 200
        assert response.json()["feedback"] == "positive"

    @pytest.mark.asyncio
    async def test_feedback_message_not_found(
        self,
        conv_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Should return 404 for non-existent message."""
        with (
            patch("app.api.v1.conversations.SQLMessageRepository") as mock_msg_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_msg = mock_msg_cls.return_value
            mock_msg.update_feedback = AsyncMock(return_value=None)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await conv_client.patch(
                "/api/v1/conversations/messages/nonexistent/feedback",
                json={"feedback": "negative"},
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 404
