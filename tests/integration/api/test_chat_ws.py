"""Integration tests for WebSocket chat streaming endpoint.

Tests use FastAPI's WebSocket test client with mocked dependencies
to verify the full streaming flow without a real LLM or database.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from app.core.security import create_access_token
from app.domain.models.enums import UserRole
from app.domain.models.user import User
from app.main import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def test_user() -> User:
    """Authenticated test user."""
    return User(
        id="user-ws-1",
        tenant_id="tenant-ws-1",
        email="wsuser@example.com",
        password_hash="$2b$12$hashed",
        role=UserRole.VIEWER,
    )


@pytest.fixture
def valid_token() -> str:
    """Valid JWT access token for test user."""
    return create_access_token(
        user_id="user-ws-1",
        tenant_id="tenant-ws-1",
        role="viewer",
        secret_key="change-me-to-another-random-secret",
    )


@pytest.fixture
def mock_session() -> AsyncMock:
    """Mock async database session."""
    session = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def app_with_mocks(mock_session: AsyncMock, test_user: User) -> MagicMock:
    """Create app with mocked DB and chat service on app.state."""
    app = create_app()

    async def _mock_session_gen() -> AsyncGenerator[AsyncMock, None]:
        yield mock_session

    from app.infrastructure.database.connection import get_async_session

    app.dependency_overrides[get_async_session] = _mock_session_gen

    # Mock the chat service on app.state
    mock_chat_service = AsyncMock()
    app.state.chat_service = mock_chat_service

    # Mock the connection manager on app.state
    from app.infrastructure.websocket.connection_manager import ConnectionManager

    app.state.ws_manager = ConnectionManager()

    return app


@pytest.fixture
def ws_client(app_with_mocks: MagicMock) -> TestClient:
    """Synchronous test client for WebSocket testing."""
    return TestClient(app_with_mocks)


# ── WebSocket Connection Tests ──────────────────────────────────


class TestWebSocketConnect:
    """Tests for WebSocket handshake and authentication."""

    def test_connect_with_valid_token(
        self,
        ws_client: TestClient,
        valid_token: str,
        test_user: User,
    ) -> None:
        """Valid JWT token should allow WebSocket connection."""
        with patch(
            "app.api.v1.chat_ws.SQLUserRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=test_user)

            with ws_client.websocket_connect(
                f"/api/v1/ws/chat?token={valid_token}"
            ) as ws:
                # Connection established — send a close to clean up
                ws.close()

    def test_connect_without_token_rejected(
        self,
        ws_client: TestClient,
    ) -> None:
        """Missing token should reject the WebSocket connection."""
        with pytest.raises(Exception):  # noqa: B017
            with ws_client.websocket_connect("/api/v1/ws/chat") as ws:
                ws.close()

    def test_connect_with_invalid_token_rejected(
        self,
        ws_client: TestClient,
    ) -> None:
        """Invalid JWT token should reject the WebSocket connection."""
        with pytest.raises(Exception):  # noqa: B017
            with ws_client.websocket_connect(
                "/api/v1/ws/chat?token=invalid.jwt.token"
            ) as ws:
                ws.close()

    def test_connect_with_expired_token_rejected(
        self,
        ws_client: TestClient,
    ) -> None:
        """Expired JWT token should reject the WebSocket connection."""
        expired_token = create_access_token(
            user_id="user-ws-1",
            tenant_id="tenant-ws-1",
            role="viewer",
            secret_key="change-me-to-another-random-secret",
            expires_minutes=-1,  # Already expired
        )
        with pytest.raises(Exception):  # noqa: B017
            with ws_client.websocket_connect(
                f"/api/v1/ws/chat?token={expired_token}"
            ) as ws:
                ws.close()

    def test_connect_with_deleted_user_rejected(
        self,
        ws_client: TestClient,
        valid_token: str,
    ) -> None:
        """Valid JWT for a deleted user should reject the connection."""
        with patch(
            "app.api.v1.chat_ws.SQLUserRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=None)

            with pytest.raises(Exception):  # noqa: B017
                with ws_client.websocket_connect(
                    f"/api/v1/ws/chat?token={valid_token}"
                ) as ws:
                    ws.close()


# ── WebSocket Streaming Tests ───────────────────────────────────


class TestWebSocketStreaming:
    """Tests for streaming chat responses via WebSocket."""

    def test_send_message_receives_streaming_tokens(
        self,
        ws_client: TestClient,
        valid_token: str,
        test_user: User,
        app_with_mocks: MagicMock,
    ) -> None:
        """Sending a message should produce token + done frames."""

        async def _mock_stream(message: str, tenant_id: str, conversation_id: str | None = None):
            """Mock streaming that yields tokens."""
            yield {"type": "token", "data": "Hello "}
            yield {"type": "token", "data": "world!"}
            yield {
                "type": "done",
                "data": {
                    "conversation_id": "conv-123",
                    "model_used": "test-model",
                    "sources": [],
                },
            }

        app_with_mocks.state.chat_service.stream_message = _mock_stream

        with patch(
            "app.api.v1.chat_ws.SQLUserRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=test_user)

            with ws_client.websocket_connect(
                f"/api/v1/ws/chat?token={valid_token}"
            ) as ws:
                ws.send_json({"message": "Hello?"})

                # Should receive token frames
                frame1 = ws.receive_json()
                assert frame1["type"] == "token"
                assert frame1["data"] == "Hello "

                frame2 = ws.receive_json()
                assert frame2["type"] == "token"
                assert frame2["data"] == "world!"

                # Should receive done frame
                done_frame = ws.receive_json()
                assert done_frame["type"] == "done"
                assert done_frame["data"]["conversation_id"] == "conv-123"

    def test_tenant_id_derived_from_jwt(
        self,
        ws_client: TestClient,
        valid_token: str,
        test_user: User,
        app_with_mocks: MagicMock,
    ) -> None:
        """stream_message must be called with the JWT user's tenant_id.

        This is a cross-tenant isolation regression test: the tenant_id
        passed to stream_message must come from the authenticated user's
        JWT, not from any client-supplied value in the WebSocket message.
        """
        captured_calls: list[dict[str, str]] = []

        async def _capturing_stream(
            message: str,
            tenant_id: str,
            conversation_id: str | None = None,
        ):
            captured_calls.append({"tenant_id": tenant_id})
            yield {"type": "done", "data": {"conversation_id": "c1"}}

        app_with_mocks.state.chat_service.stream_message = _capturing_stream

        with patch(
            "app.api.v1.chat_ws.SQLUserRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=test_user)

            with ws_client.websocket_connect(
                f"/api/v1/ws/chat?token={valid_token}"
            ) as ws:
                # Client sends a message — tenant_id should NOT come from here
                ws.send_json({"message": "Hello from attacker"})
                ws.receive_json()  # consume done frame

        # Verify tenant_id came from JWT user (tenant-ws-1), not from client
        assert len(captured_calls) == 1
        assert captured_calls[0]["tenant_id"] == "tenant-ws-1"

    def test_send_empty_message_receives_error(
        self,
        ws_client: TestClient,
        valid_token: str,
        test_user: User,
    ) -> None:
        """Empty message should produce an error frame."""
        with patch(
            "app.api.v1.chat_ws.SQLUserRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=test_user)

            with ws_client.websocket_connect(
                f"/api/v1/ws/chat?token={valid_token}"
            ) as ws:
                ws.send_json({"message": ""})

                frame = ws.receive_json()
                assert frame["type"] == "error"

    def test_llm_error_produces_error_frame(
        self,
        ws_client: TestClient,
        valid_token: str,
        test_user: User,
        app_with_mocks: MagicMock,
    ) -> None:
        """LLM failure during streaming should produce an error frame."""
        from app.core.exceptions import LLMError

        async def _mock_stream_error(message: str, tenant_id: str, conversation_id: str | None = None):
            yield {"type": "token", "data": "partial"}
            raise LLMError("Connection lost")

        app_with_mocks.state.chat_service.stream_message = _mock_stream_error

        with patch(
            "app.api.v1.chat_ws.SQLUserRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=test_user)

            with ws_client.websocket_connect(
                f"/api/v1/ws/chat?token={valid_token}"
            ) as ws:
                ws.send_json({"message": "Test query"})

                # First frame is partial token
                frame1 = ws.receive_json()
                assert frame1["type"] == "token"

                # Next frame should be error
                frame2 = ws.receive_json()
                assert frame2["type"] == "error"
                assert "Connection lost" in frame2["data"]["message"]
