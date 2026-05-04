"""Tests for chat schemas, service, and endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from app.api.v1.chat_service import ChatService
from app.api.v1.schemas import ChatRequest, ChatResponse, SourceCitation
from app.main import create_app


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

        with patch("app.api.v1.chat_service.run_rag_pipeline", new_callable=AsyncMock) as mock_rag:
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

        with patch("app.api.v1.chat_service.run_rag_pipeline", new_callable=AsyncMock) as mock_rag:
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


class TestChatEndpoint:
    """Test suite for POST /api/v1/chat endpoint."""

    def test_chat_missing_tenant_header_returns_422(self) -> None:
        """Missing X-Tenant-ID should return 422."""
        app = create_app()
        client = TestClient(app)
        response = client.post("/api/v1/chat", json={"message": "Hello"})
        assert response.status_code == 422

    def test_chat_empty_message_returns_422(self) -> None:
        """Empty message should return 422 validation error."""
        app = create_app()
        client = TestClient(app)
        response = client.post(
            "/api/v1/chat",
            json={"message": ""},
            headers={"X-Tenant-ID": "tenant-1"},
        )
        assert response.status_code == 422

    def test_chat_success(self) -> None:
        """Valid request should return 200 with ChatResponse."""
        app = create_app()
        client = TestClient(app)

        with patch("app.api.v1.chat_router._build_chat_service") as mock_build:
            mock_service = AsyncMock()
            mock_service.process_message.return_value = {
                "answer": "Test answer",
                "conversation_id": "conv-123",
                "sources": [{"content": "doc text", "score": 0.9, "id": "doc-1"}],
                "escalated": False,
                "escalation_reason": "",
                "model_used": "test-model",
            }
            mock_build.return_value = mock_service

            response = client.post(
                "/api/v1/chat",
                json={"message": "How do I reset my password?"},
                headers={"X-Tenant-ID": "tenant-123"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "Test answer"
            assert data["conversation_id"] == "conv-123"
            assert len(data["sources"]) == 1
            assert data["escalated"] is False

    def test_chat_escalation(self) -> None:
        """Escalated queries should have escalated=True."""
        app = create_app()
        client = TestClient(app)

        with patch("app.api.v1.chat_router._build_chat_service") as mock_build:
            mock_service = AsyncMock()
            mock_service.process_message.return_value = {
                "answer": "Escalating to human agent.",
                "conversation_id": "conv-456",
                "sources": [],
                "escalated": True,
                "escalation_reason": "No relevant docs",
                "model_used": "",
            }
            mock_build.return_value = mock_service

            response = client.post(
                "/api/v1/chat",
                json={"message": "Something obscure"},
                headers={"X-Tenant-ID": "tenant-123"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["escalated"] is True
