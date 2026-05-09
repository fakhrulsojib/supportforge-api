"""Tests for chat schemas, service, and REST endpoint.

Migrated to canonical imports and JWT-authenticated endpoint tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from app.api.schemas.chat import ChatRequest, ChatResponse, SourceCitation
from app.core.security import create_access_token
from app.domain.models.enums import UserRole
from app.domain.models.user import User
from app.domain.services.chat_service import ChatService
from app.main import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def test_user() -> User:
    """Authenticated test user."""
    return User(
        id="user-chat-1",
        tenant_id="tenant-chat-1",
        email="chatuser@example.com",
        password_hash="$2b$12$hashed",
        role=UserRole.VIEWER,
    )


@pytest.fixture
def valid_token() -> str:
    """Valid JWT access token for test user."""
    return create_access_token(
        user_id="user-chat-1",
        tenant_id="tenant-chat-1",
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
def app_with_mocks(mock_session: AsyncMock, test_user: User) -> TestClient:
    """Create app with mocked DB, auth, and chat service."""
    app = create_app()

    async def _mock_session_gen() -> AsyncGenerator[AsyncMock, None]:
        yield mock_session

    from app.infrastructure.database.connection import get_async_session

    app.dependency_overrides[get_async_session] = _mock_session_gen

    # Mock chat service on app.state
    mock_chat_service = AsyncMock()
    app.state.chat_service = mock_chat_service

    return app


# ── Schema Tests ────────────────────────────────────────────────


class TestChatRequest:
    """Test suite for ChatRequest schema."""

    def test_valid_request(self) -> None:
        req = ChatRequest(message="Hello!")
        assert req.message == "Hello!"
        assert req.conversation_id is None

    def test_with_conversation_id(self) -> None:
        req = ChatRequest(message="Follow-up", conversation_id="conv-123")
        assert req.conversation_id == "conv-123"

    def test_empty_message_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(message="")


class TestChatResponse:
    """Test suite for ChatResponse schema."""

    def test_valid_response(self) -> None:
        resp = ChatResponse(answer="Hello!", conversation_id="conv-123")
        assert resp.answer == "Hello!"
        assert resp.escalated is False

    def test_response_with_sources(self) -> None:
        source = SourceCitation(content="Test doc", score=0.9, id="doc-1")
        resp = ChatResponse(answer="Answer", conversation_id="conv-1", sources=[source])
        assert len(resp.sources) == 1


# ── Service Tests ───────────────────────────────────────────────


class TestChatService:
    """Test suite for ChatService."""

    @pytest.mark.asyncio
    async def test_process_message_success(self) -> None:
        """Service should run RAG pipeline and return result."""
        llm_provider = AsyncMock()
        vector_store = AsyncMock()
        embedding_service = AsyncMock()

        service = ChatService(
            llm_provider=llm_provider,
            vector_store=vector_store,
            embedding_service=embedding_service,
        )

        with patch("app.domain.services.chat_service.run_rag_pipeline", new_callable=AsyncMock) as mock_rag:
            mock_rag.return_value = {
                "answer": "Test answer",
                "sources": [{"content": "source", "score": 0.8, "id": "1"}],
                "should_escalate": False,
                "escalation_reason": "",
                "model_used": "test-model",
            }

            result = await service.process_message(
                message="Hello",
                tenant_id="tenant-1",
            )

            assert result["answer"] == "Test answer"
            assert result["conversation_id"]  # Should be a UUID
            mock_rag.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_message_preserves_conversation_id(self) -> None:
        """Should use provided conversation_id."""
        service = ChatService(
            llm_provider=AsyncMock(),
            vector_store=AsyncMock(),
            embedding_service=AsyncMock(),
        )

        with patch("app.domain.services.chat_service.run_rag_pipeline", new_callable=AsyncMock) as mock_rag:
            mock_rag.return_value = {
                "answer": "Reply",
                "sources": [],
                "should_escalate": False,
                "escalation_reason": "",
                "model_used": "",
            }

            result = await service.process_message(
                message="Follow up",
                tenant_id="tenant-1",
                conversation_id="existing-conv-123",
            )

            assert result["conversation_id"] == "existing-conv-123"


class TestChatServiceStreaming:
    """Test suite for ChatService.stream_message()."""

    @pytest.mark.asyncio
    async def test_stream_message_yields_source_token_done_frames(self) -> None:
        """stream_message should yield source, token, and done frames."""
        mock_llm = AsyncMock()
        mock_llm.default_model = "test-model"

        async def _mock_stream(*args, **kwargs):
            yield {"type": "content", "text": "Hello "}
            yield {"type": "content", "text": "world!"}

        mock_llm.stream = _mock_stream
        mock_vs = AsyncMock()
        mock_embed = AsyncMock()

        service = ChatService(
            llm_provider=mock_llm,
            vector_store=mock_vs,
            embedding_service=mock_embed,
        )

        with (
            patch("app.domain.services.chat_service.retrieve_node", new_callable=AsyncMock) as mock_retrieve,
            patch("app.domain.services.chat_service.grade_node", new_callable=AsyncMock) as mock_grade,
        ):
            mock_retrieve.return_value = {
                "query": "test",
                "tenant_id": "t1",
                "retrieved_docs": [{"content": "doc text", "score": 0.9, "id": "d1"}],
                "relevant_docs": [],
                "answer": "",
                "sources": [],
                "should_escalate": False,
                "escalation_reason": "",
                "model_used": "",
                "tokens_in": 0,
                "tokens_out": 0,
            }
            mock_grade.return_value = {
                "query": "test",
                "tenant_id": "t1",
                "retrieved_docs": [{"content": "doc text", "score": 0.9, "id": "d1"}],
                "relevant_docs": [{"content": "doc text", "score": 0.9, "id": "d1"}],
                "answer": "",
                "sources": [],
                "should_escalate": False,
                "escalation_reason": "",
                "model_used": "",
                "tokens_in": 0,
                "tokens_out": 0,
            }

            frames = []
            async for frame in service.stream_message(
                message="test query", tenant_id="t1"
            ):
                frames.append(frame)

        # Should have: 1 source + 2 tokens + 1 done = 4 frames
        assert len(frames) == 4
        assert frames[0]["type"] == "source"
        assert frames[0]["data"]["id"] == "d1"
        assert frames[1]["type"] == "token"
        assert frames[1]["data"] == "Hello "
        assert frames[2]["type"] == "token"
        assert frames[2]["data"] == "world!"
        assert frames[3]["type"] == "done"
        assert frames[3]["data"]["model_used"] == "test-model"
        assert frames[3]["data"]["conversation_id"]  # UUID generated

    @pytest.mark.asyncio
    async def test_stream_message_escalation_path(self) -> None:
        """When should_escalate is True, yield escalation frames."""
        service = ChatService(
            llm_provider=AsyncMock(),
            vector_store=AsyncMock(),
            embedding_service=AsyncMock(),
        )

        with (
            patch("app.domain.services.chat_service.retrieve_node", new_callable=AsyncMock) as mock_retrieve,
            patch("app.domain.services.chat_service.grade_node", new_callable=AsyncMock) as mock_grade,
            patch("app.domain.services.chat_service.escalation_node", new_callable=AsyncMock) as mock_esc,
        ):
            mock_retrieve.return_value = {
                "query": "test",
                "tenant_id": "t1",
                "retrieved_docs": [],
                "relevant_docs": [],
                "answer": "",
                "sources": [],
                "should_escalate": False,
                "escalation_reason": "",
                "model_used": "",
                "tokens_in": 0,
                "tokens_out": 0,
            }
            mock_grade.return_value = {
                "query": "test",
                "tenant_id": "t1",
                "retrieved_docs": [],
                "relevant_docs": [],
                "answer": "",
                "sources": [],
                "should_escalate": True,
                "escalation_reason": "No docs found",
                "model_used": "",
                "tokens_in": 0,
                "tokens_out": 0,
            }
            mock_esc.return_value = {
                "query": "test",
                "tenant_id": "t1",
                "retrieved_docs": [],
                "relevant_docs": [],
                "answer": "Escalating to human agent.",
                "sources": [],
                "should_escalate": True,
                "escalation_reason": "No docs found",
                "model_used": "",
                "tokens_in": 0,
                "tokens_out": 0,
            }

            frames = []
            async for frame in service.stream_message(
                message="test query", tenant_id="t1"
            ):
                frames.append(frame)

        # Should have: 1 token (escalation message) + 1 done = 2 frames
        assert len(frames) == 2
        assert frames[0]["type"] == "token"
        assert "Escalating" in frames[0]["data"]
        assert frames[1]["type"] == "done"
        assert frames[1]["data"]["escalated"] is True
        assert frames[1]["data"]["escalation_reason"] == "No docs found"

    @pytest.mark.asyncio
    async def test_stream_message_preserves_conversation_id(self) -> None:
        """stream_message should use provided conversation_id."""
        mock_llm = AsyncMock()
        mock_llm.default_model = "m"

        async def _empty_stream(*args, **kwargs):
            return
            yield  # Make it a generator  # type: ignore[misc]  # noqa: E501

        mock_llm.stream = _empty_stream

        service = ChatService(
            llm_provider=mock_llm,
            vector_store=AsyncMock(),
            embedding_service=AsyncMock(),
        )

        with (
            patch("app.domain.services.chat_service.retrieve_node", new_callable=AsyncMock) as mock_retrieve,
            patch("app.domain.services.chat_service.grade_node", new_callable=AsyncMock) as mock_grade,
        ):
            mock_retrieve.return_value = {
                "query": "test", "tenant_id": "t1", "retrieved_docs": [],
                "relevant_docs": [], "answer": "", "sources": [],
                "should_escalate": False, "escalation_reason": "",
                "model_used": "", "tokens_in": 0, "tokens_out": 0,
            }
            mock_grade.return_value = {
                "query": "test", "tenant_id": "t1", "retrieved_docs": [],
                "relevant_docs": [{"content": "x", "score": 0.5, "id": "1"}],
                "answer": "", "sources": [],
                "should_escalate": False, "escalation_reason": "",
                "model_used": "", "tokens_in": 0, "tokens_out": 0,
            }

            frames = []
            async for frame in service.stream_message(
                message="test", tenant_id="t1", conversation_id="my-conv-42"
            ):
                frames.append(frame)

        done_frame = [f for f in frames if f["type"] == "done"][0]
        assert done_frame["data"]["conversation_id"] == "my-conv-42"


class TestChatServiceThinking:
    """Test suite for thinking frame support in stream_message()."""

    @staticmethod
    def _make_rag_state(*, with_docs: bool = True) -> tuple[dict, dict]:
        """Helper to create retrieve and grade mock return values."""
        doc = {"content": "doc text", "score": 0.9, "id": "d1"}
        base = {
            "query": "test", "tenant_id": "t1",
            "retrieved_docs": [doc] if with_docs else [],
            "relevant_docs": [doc] if with_docs else [],
            "answer": "", "sources": [],
            "should_escalate": False, "escalation_reason": "",
            "model_used": "", "tokens_in": 0, "tokens_out": 0,
        }
        return base, base.copy()

    @pytest.mark.asyncio
    async def test_stream_yields_thinking_and_content_frames(self) -> None:
        """Thinking tokens yield 'thinking' frames; content tokens yield 'token' frames."""
        mock_llm = AsyncMock()
        mock_llm.default_model = "qwen3:4b"

        async def _mock_stream(*args, **kwargs):
            yield {"type": "thinking", "text": "Let me reason..."}
            yield {"type": "thinking", "text": " about this."}
            yield {"type": "content", "text": "The answer is 42."}

        mock_llm.stream = _mock_stream
        service = ChatService(
            llm_provider=mock_llm,
            vector_store=AsyncMock(),
            embedding_service=AsyncMock(),
        )
        retrieve_state, grade_state = self._make_rag_state()

        with (
            patch("app.domain.services.chat_service.retrieve_node", new_callable=AsyncMock) as mr,
            patch("app.domain.services.chat_service.grade_node", new_callable=AsyncMock) as mg,
        ):
            mr.return_value = retrieve_state
            mg.return_value = grade_state

            frames = []
            async for f in service.stream_message(message="test", tenant_id="t1"):
                frames.append(f)

        thinking_frames = [f for f in frames if f["type"] == "thinking"]
        token_frames = [f for f in frames if f["type"] == "token"]
        done_frames = [f for f in frames if f["type"] == "done"]

        assert len(thinking_frames) == 2
        assert thinking_frames[0]["data"] == "Let me reason..."
        assert thinking_frames[1]["data"] == " about this."
        assert len(token_frames) == 1
        assert token_frames[0]["data"] == "The answer is 42."
        assert len(done_frames) == 1
        assert done_frames[0]["data"]["thinking_text"] == "Let me reason... about this."

    @pytest.mark.asyncio
    async def test_stream_no_thinking_yields_empty_thinking_text(self) -> None:
        """Models without thinking should yield empty thinking_text in done frame."""
        mock_llm = AsyncMock()
        mock_llm.default_model = "llama3"

        async def _mock_stream(*args, **kwargs):
            yield {"type": "content", "text": "Direct answer."}

        mock_llm.stream = _mock_stream
        service = ChatService(
            llm_provider=mock_llm,
            vector_store=AsyncMock(),
            embedding_service=AsyncMock(),
        )
        retrieve_state, grade_state = self._make_rag_state()

        with (
            patch("app.domain.services.chat_service.retrieve_node", new_callable=AsyncMock) as mr,
            patch("app.domain.services.chat_service.grade_node", new_callable=AsyncMock) as mg,
        ):
            mr.return_value = retrieve_state
            mg.return_value = grade_state

            frames = []
            async for f in service.stream_message(message="test", tenant_id="t1"):
                frames.append(f)

        thinking_frames = [f for f in frames if f["type"] == "thinking"]
        done_frame = [f for f in frames if f["type"] == "done"][0]
        assert len(thinking_frames) == 0
        assert done_frame["data"]["thinking_text"] == ""

    @pytest.mark.asyncio
    async def test_stream_thinking_only_no_content(self) -> None:
        """If model only produces thinking but no content, done frame should still emit."""
        mock_llm = AsyncMock()
        mock_llm.default_model = "qwen3:4b"

        async def _mock_stream(*args, **kwargs):
            yield {"type": "thinking", "text": "Internal reasoning only."}

        mock_llm.stream = _mock_stream
        service = ChatService(
            llm_provider=mock_llm,
            vector_store=AsyncMock(),
            embedding_service=AsyncMock(),
        )
        retrieve_state, grade_state = self._make_rag_state()

        with (
            patch("app.domain.services.chat_service.retrieve_node", new_callable=AsyncMock) as mr,
            patch("app.domain.services.chat_service.grade_node", new_callable=AsyncMock) as mg,
        ):
            mr.return_value = retrieve_state
            mg.return_value = grade_state

            frames = []
            async for f in service.stream_message(message="test", tenant_id="t1"):
                frames.append(f)

        token_frames = [f for f in frames if f["type"] == "token"]
        done_frame = [f for f in frames if f["type"] == "done"][0]
        assert len(token_frames) == 0
        assert done_frame["data"]["thinking_text"] == "Internal reasoning only."

    @pytest.mark.asyncio
    async def test_stream_backward_compat_string_tokens(self) -> None:
        """Plain string yields (non-dict) should still work as content frames."""
        mock_llm = AsyncMock()
        mock_llm.default_model = "legacy-model"

        async def _mock_stream(*args, **kwargs):
            yield "plain string token"

        mock_llm.stream = _mock_stream
        service = ChatService(
            llm_provider=mock_llm,
            vector_store=AsyncMock(),
            embedding_service=AsyncMock(),
        )
        retrieve_state, grade_state = self._make_rag_state()

        with (
            patch("app.domain.services.chat_service.retrieve_node", new_callable=AsyncMock) as mr,
            patch("app.domain.services.chat_service.grade_node", new_callable=AsyncMock) as mg,
        ):
            mr.return_value = retrieve_state
            mg.return_value = grade_state

            frames = []
            async for f in service.stream_message(message="test", tenant_id="t1"):
                frames.append(f)

        token_frames = [f for f in frames if f["type"] == "token"]
        assert len(token_frames) == 1
        assert token_frames[0]["data"] == "plain string token"

class TestChatEndpoint:
    """Test suite for POST /api/v1/chat endpoint (JWT-protected)."""

    def test_chat_missing_auth_returns_401(self) -> None:
        """Missing Authorization header should return 401."""
        app = create_app()
        client = TestClient(app)
        response = client.post("/api/v1/chat", json={"message": "Hello"})
        assert response.status_code == 401

    def test_chat_invalid_token_returns_401(self) -> None:
        """Invalid JWT token should return 401."""
        app = create_app()
        client = TestClient(app)
        response = client.post(
            "/api/v1/chat",
            json={"message": "Hello"},
            headers={"Authorization": "Bearer invalid.jwt.token"},
        )
        assert response.status_code == 401

    def test_chat_empty_message_returns_422(
        self,
        app_with_mocks: TestClient,
        valid_token: str,
        test_user: User,
    ) -> None:
        """Empty message should return 422 validation error."""
        client = TestClient(app_with_mocks)

        with patch(
            "app.core.dependencies.SQLUserRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=test_user)

            response = client.post(
                "/api/v1/chat",
                json={"message": ""},
                headers={"Authorization": f"Bearer {valid_token}"},
            )
        assert response.status_code == 422

    def test_chat_success(
        self,
        app_with_mocks: TestClient,
        valid_token: str,
        test_user: User,
    ) -> None:
        """Valid authenticated request should return 200 with ChatResponse."""
        app_with_mocks.state.chat_service.process_message = AsyncMock(
            return_value={
                "answer": "Test answer",
                "conversation_id": "conv-123",
                "sources": [{"content": "doc text", "score": 0.9, "id": "doc-1"}],
                "escalated": False,
                "escalation_reason": "",
                "model_used": "test-model",
            }
        )
        client = TestClient(app_with_mocks)

        with patch(
            "app.core.dependencies.SQLUserRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=test_user)

            response = client.post(
                "/api/v1/chat",
                json={"message": "How do I reset my password?"},
                headers={"Authorization": f"Bearer {valid_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer"
        assert data["conversation_id"] == "conv-123"
        assert len(data["sources"]) == 1
        assert data["escalated"] is False

    def test_chat_escalation(
        self,
        app_with_mocks: TestClient,
        valid_token: str,
        test_user: User,
    ) -> None:
        """Escalated queries should have escalated=True."""
        app_with_mocks.state.chat_service.process_message = AsyncMock(
            return_value={
                "answer": "Escalating to human agent.",
                "conversation_id": "conv-456",
                "sources": [],
                "escalated": True,
                "escalation_reason": "No relevant docs",
                "model_used": "",
            }
        )
        client = TestClient(app_with_mocks)

        with patch(
            "app.core.dependencies.SQLUserRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=test_user)

            response = client.post(
                "/api/v1/chat",
                json={"message": "Something obscure"},
                headers={"Authorization": f"Bearer {valid_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["escalated"] is True

    def test_chat_derives_tenant_from_jwt(
        self,
        app_with_mocks: TestClient,
        valid_token: str,
        test_user: User,
    ) -> None:
        """Tenant ID should be derived from the JWT user, not from a header."""
        app_with_mocks.state.chat_service.process_message = AsyncMock(
            return_value={
                "answer": "OK",
                "conversation_id": "conv-789",
                "sources": [],
                "escalated": False,
                "escalation_reason": "",
                "model_used": "",
            }
        )
        client = TestClient(app_with_mocks)

        with patch(
            "app.core.dependencies.SQLUserRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=test_user)

            response = client.post(
                "/api/v1/chat",
                json={"message": "Hello"},
                headers={"Authorization": f"Bearer {valid_token}"},
            )

        assert response.status_code == 200
        # Verify process_message was called with the JWT user's tenant_id
        call_kwargs = app_with_mocks.state.chat_service.process_message.call_args
        assert call_kwargs.kwargs.get("tenant_id") == "tenant-chat-1" or \
            call_kwargs[1].get("tenant_id") == "tenant-chat-1"


class TestChatServiceOutputValidation:
    """Test suite for output validation integration in stream_message()."""

    @staticmethod
    def _make_rag_state(*, with_docs: bool = True) -> tuple[dict, dict]:
        """Helper to create retrieve and grade mock return values."""
        doc = {"content": "doc text about orders", "score": 0.9, "id": "d1"}
        base = {
            "query": "test", "tenant_id": "t1",
            "retrieved_docs": [doc] if with_docs else [],
            "relevant_docs": [doc] if with_docs else [],
            "answer": "", "sources": [],
            "should_escalate": False, "escalation_reason": "",
            "model_used": "", "tokens_in": 0, "tokens_out": 0,
        }
        return base, base.copy()

    @pytest.mark.asyncio
    async def test_stream_message_clean_output_passes_validation(self) -> None:
        """Clean LLM output should yield validation_status=passed in done frame."""
        mock_llm = AsyncMock()
        mock_llm.default_model = "test-model"

        async def _mock_stream(*args, **kwargs):
            yield {"type": "content", "text": "Your order will arrive soon."}

        mock_llm.stream = _mock_stream
        service = ChatService(
            llm_provider=mock_llm,
            vector_store=AsyncMock(),
            embedding_service=AsyncMock(),
        )
        retrieve_state, grade_state = self._make_rag_state()

        with (
            patch("app.domain.services.chat_service.retrieve_node", new_callable=AsyncMock) as mr,
            patch("app.domain.services.chat_service.grade_node", new_callable=AsyncMock) as mg,
        ):
            mr.return_value = retrieve_state
            mg.return_value = grade_state

            frames = []
            async for f in service.stream_message(message="test", tenant_id="t1"):
                frames.append(f)

        done_frame = [f for f in frames if f["type"] == "done"][0]
        assert done_frame["data"]["validation_status"] == "passed"
        # No disclaimer frame should be emitted for clean responses
        disclaimer_frames = [f for f in frames if f["type"] == "disclaimer"]
        assert len(disclaimer_frames) == 0

    @pytest.mark.asyncio
    async def test_stream_message_fabricated_phone_flags_validation(self) -> None:
        """Fabricated phone number in LLM output should yield validation_status=flagged."""
        mock_llm = AsyncMock()
        mock_llm.default_model = "test-model"

        async def _mock_stream(*args, **kwargs):
            yield {"type": "content", "text": "Call us at 555-999-8888 for help."}

        mock_llm.stream = _mock_stream
        service = ChatService(
            llm_provider=mock_llm,
            vector_store=AsyncMock(),
            embedding_service=AsyncMock(),
        )
        retrieve_state, grade_state = self._make_rag_state()

        with (
            patch("app.domain.services.chat_service.retrieve_node", new_callable=AsyncMock) as mr,
            patch("app.domain.services.chat_service.grade_node", new_callable=AsyncMock) as mg,
        ):
            mr.return_value = retrieve_state
            mg.return_value = grade_state

            frames = []
            async for f in service.stream_message(message="test", tenant_id="t1"):
                frames.append(f)

        done_frame = [f for f in frames if f["type"] == "done"][0]
        assert done_frame["data"]["validation_status"] == "flagged"
        # Verify disclaimer frame is emitted to client (M-1 fix)
        disclaimer_frames = [f for f in frames if f["type"] == "disclaimer"]
        assert len(disclaimer_frames) == 1
        assert "could not be verified" in disclaimer_frames[0]["data"]

    @pytest.mark.asyncio
    async def test_stream_message_validation_logs_on_failure(self) -> None:
        """Validation failures should emit structured log warnings."""
        mock_llm = AsyncMock()
        mock_llm.default_model = "test-model"

        async def _mock_stream(*args, **kwargs):
            yield {"type": "content", "text": "Email us at fake@hallucinated.com."}

        mock_llm.stream = _mock_stream
        service = ChatService(
            llm_provider=mock_llm,
            vector_store=AsyncMock(),
            embedding_service=AsyncMock(),
        )
        retrieve_state, grade_state = self._make_rag_state()

        with (
            patch("app.domain.services.chat_service.retrieve_node", new_callable=AsyncMock) as mr,
            patch("app.domain.services.chat_service.grade_node", new_callable=AsyncMock) as mg,
            patch("app.domain.services.chat_service.logger") as mock_logger,
        ):
            mr.return_value = retrieve_state
            mg.return_value = grade_state

            frames = []
            async for f in service.stream_message(message="test", tenant_id="t1"):
                frames.append(f)

        # Should have called logger.warning with output_validation_failed
        warning_calls = mock_logger.warning.call_args_list
        validation_warnings = [
            c for c in warning_calls
            if c.args and c.args[0] == "output_validation_failed"
        ]
        assert len(validation_warnings) >= 1
        # Check structured metadata
        assert validation_warnings[0].kwargs["rule_violated"] == "fabricated_email"
        assert "fake@hallucinated.com" in validation_warnings[0].kwargs["snippet"]

    @pytest.mark.asyncio
    async def test_stream_message_context_phone_not_flagged(self) -> None:
        """Phone number present in retrieved context should NOT be flagged."""
        mock_llm = AsyncMock()
        mock_llm.default_model = "test-model"

        async def _mock_stream(*args, **kwargs):
            yield {"type": "content", "text": "Call us at 800-555-1234."}

        mock_llm.stream = _mock_stream
        service = ChatService(
            llm_provider=mock_llm,
            vector_store=AsyncMock(),
            embedding_service=AsyncMock(),
        )
        # Context contains the same phone number
        doc = {
            "content": "For support, call 800-555-1234.",
            "score": 0.9,
            "id": "d1",
        }
        retrieve_state = {
            "query": "test", "tenant_id": "t1",
            "retrieved_docs": [doc], "relevant_docs": [doc],
            "answer": "", "sources": [],
            "should_escalate": False, "escalation_reason": "",
            "model_used": "", "tokens_in": 0, "tokens_out": 0,
        }

        with (
            patch("app.domain.services.chat_service.retrieve_node", new_callable=AsyncMock) as mr,
            patch("app.domain.services.chat_service.grade_node", new_callable=AsyncMock) as mg,
        ):
            mr.return_value = retrieve_state
            mg.return_value = retrieve_state.copy()

            frames = []
            async for f in service.stream_message(message="test", tenant_id="t1"):
                frames.append(f)

        done_frame = [f for f in frames if f["type"] == "done"][0]
        assert done_frame["data"]["validation_status"] == "passed"

    @pytest.mark.asyncio
    async def test_stream_message_forbidden_latex_always_flagged(self) -> None:
        """LaTeX patterns should be flagged even if context contains them."""
        mock_llm = AsyncMock()
        mock_llm.default_model = "test-model"

        async def _mock_stream(*args, **kwargs):
            yield {"type": "content", "text": "The answer is \\boxed{42}."}

        mock_llm.stream = _mock_stream
        service = ChatService(
            llm_provider=mock_llm,
            vector_store=AsyncMock(),
            embedding_service=AsyncMock(),
        )
        retrieve_state, grade_state = self._make_rag_state()

        with (
            patch("app.domain.services.chat_service.retrieve_node", new_callable=AsyncMock) as mr,
            patch("app.domain.services.chat_service.grade_node", new_callable=AsyncMock) as mg,
        ):
            mr.return_value = retrieve_state
            mg.return_value = grade_state

            frames = []
            async for f in service.stream_message(message="test", tenant_id="t1"):
                frames.append(f)

        done_frame = [f for f in frames if f["type"] == "done"][0]
        assert done_frame["data"]["validation_status"] == "flagged"

    @pytest.mark.asyncio
    async def test_stream_message_escalation_skips_validation(self) -> None:
        """Escalated responses bypass the validation pipeline."""
        service = ChatService(
            llm_provider=AsyncMock(),
            vector_store=AsyncMock(),
            embedding_service=AsyncMock(),
        )

        with (
            patch("app.domain.services.chat_service.retrieve_node", new_callable=AsyncMock) as mock_retrieve,
            patch("app.domain.services.chat_service.grade_node", new_callable=AsyncMock) as mock_grade,
            patch("app.domain.services.chat_service.escalation_node", new_callable=AsyncMock) as mock_esc,
        ):
            mock_retrieve.return_value = {
                "query": "test", "tenant_id": "t1",
                "retrieved_docs": [], "relevant_docs": [],
                "answer": "", "sources": [],
                "should_escalate": False, "escalation_reason": "",
                "model_used": "", "tokens_in": 0, "tokens_out": 0,
            }
            mock_grade.return_value = {
                "query": "test", "tenant_id": "t1",
                "retrieved_docs": [], "relevant_docs": [],
                "answer": "", "sources": [],
                "should_escalate": True, "escalation_reason": "No docs found",
                "model_used": "", "tokens_in": 0, "tokens_out": 0,
            }
            mock_esc.return_value = {
                "query": "test", "tenant_id": "t1",
                "retrieved_docs": [], "relevant_docs": [],
                "answer": "Escalating to human agent.",
                "sources": [],
                "should_escalate": True, "escalation_reason": "No docs found",
                "model_used": "", "tokens_in": 0, "tokens_out": 0,
            }

            frames = []
            async for frame in service.stream_message(
                message="test query", tenant_id="t1"
            ):
                frames.append(frame)

        done_frame = [f for f in frames if f["type"] == "done"][0]
        # Escalation path returns early without validation_status key
        assert "validation_status" not in done_frame["data"]
        assert done_frame["data"]["escalated"] is True
