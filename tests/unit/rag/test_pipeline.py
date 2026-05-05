"""Tests for the RAG pipeline nodes and orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.exceptions import LLMError
from app.domain.interfaces.vector_store import SearchResult
from app.rag.pipeline import (
    RAGState,
    escalation_node,
    generate_node,
    grade_node,
    retrieve_node,
    run_rag_pipeline,
)


def _make_state(**overrides: object) -> RAGState:
    """Create a base RAGState with defaults."""
    defaults: RAGState = {
        "query": "How do I reset my password?",
        "tenant_id": "tenant-123",
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
    defaults.update(overrides)  # type: ignore[typeddict-item]
    return defaults


def _make_search_result(content: str = "Test content", score: float = 0.8) -> SearchResult:
    """Create a SearchResult for testing."""
    return SearchResult(content=content, metadata={"source": "test.pdf"}, score=score, id="doc-1")


class TestRetrieveNode:
    """Test suite for retrieve_node."""

    @pytest.mark.asyncio
    async def test_retrieve_success(self) -> None:
        """Should embed query and search vector store."""
        vector_store = AsyncMock()
        embedding_service = AsyncMock()
        embedding_service.embed.return_value = [0.1, 0.2, 0.3]
        vector_store.search.return_value = [_make_search_result()]

        state = _make_state()
        result = await retrieve_node(state, vector_store, embedding_service)

        assert len(result["retrieved_docs"]) == 1
        assert result["retrieved_docs"][0]["content"] == "Test content"
        embedding_service.embed.assert_called_once()
        vector_store.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_empty_results(self) -> None:
        """Empty search results should return empty docs list."""
        vector_store = AsyncMock()
        embedding_service = AsyncMock()
        embedding_service.embed.return_value = [0.1, 0.2]
        vector_store.search.return_value = []

        state = _make_state()
        result = await retrieve_node(state, vector_store, embedding_service)

        assert result["retrieved_docs"] == []

    @pytest.mark.asyncio
    async def test_retrieve_embedding_error(self) -> None:
        """Embedding failures should not crash, return empty docs."""
        vector_store = AsyncMock()
        embedding_service = AsyncMock()
        embedding_service.embed.side_effect = LLMError("Connection failed")

        state = _make_state()
        result = await retrieve_node(state, vector_store, embedding_service)

        assert result["retrieved_docs"] == []


class TestGradeNode:
    """Test suite for grade_node."""

    @pytest.mark.asyncio
    async def test_grade_with_relevant_docs(self) -> None:
        """Documents above threshold should be kept."""
        llm_provider = AsyncMock()
        docs = [
            {"content": "Relevant doc", "score": 0.9, "id": "1"},
            {"content": "Irrelevant doc", "score": 0.1, "id": "2"},
        ]
        state = _make_state(retrieved_docs=docs)
        result = await grade_node(state, llm_provider, relevance_threshold=0.3)

        assert len(result["relevant_docs"]) == 1
        assert result["relevant_docs"][0]["content"] == "Relevant doc"
        assert result["should_escalate"] is False

    @pytest.mark.asyncio
    async def test_grade_no_relevant_docs_escalates(self) -> None:
        """No docs above threshold should trigger escalation."""
        llm_provider = AsyncMock()
        docs = [{"content": "Low relevance", "score": 0.1, "id": "1"}]
        state = _make_state(retrieved_docs=docs)
        result = await grade_node(state, llm_provider, relevance_threshold=0.5)

        assert result["relevant_docs"] == []
        assert result["should_escalate"] is True
        assert "threshold" in result["escalation_reason"]

    @pytest.mark.asyncio
    async def test_grade_empty_retrieved_docs_escalates(self) -> None:
        """Empty retrieved docs should trigger escalation."""
        llm_provider = AsyncMock()
        state = _make_state(retrieved_docs=[])
        result = await grade_node(state, llm_provider)

        assert result["should_escalate"] is True
        assert "No documents found" in result["escalation_reason"]

    @pytest.mark.asyncio
    async def test_grade_all_docs_above_threshold(self) -> None:
        """All docs above threshold should be kept."""
        llm_provider = AsyncMock()
        docs = [
            {"content": "Doc 1", "score": 0.8, "id": "1"},
            {"content": "Doc 2", "score": 0.7, "id": "2"},
        ]
        state = _make_state(retrieved_docs=docs)
        result = await grade_node(state, llm_provider, relevance_threshold=0.3)

        assert len(result["relevant_docs"]) == 2


class TestGenerateNode:
    """Test suite for generate_node."""

    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        """Should generate answer from context."""
        llm_provider = AsyncMock()
        llm_provider.generate.return_value = "Click Forgot Password on the login page."
        llm_provider.default_model = "test-model"

        docs = [{"content": "Reset password via the login page.", "score": 0.9, "id": "doc-1"}]
        state = _make_state(relevant_docs=docs)
        result = await generate_node(state, llm_provider)

        assert "Forgot Password" in result["answer"]
        assert len(result["sources"]) == 1
        assert result["model_used"] == "test-model"
        llm_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_llm_error_escalates(self) -> None:
        """LLM failure should escalate."""
        llm_provider = AsyncMock()
        llm_provider.generate.side_effect = LLMError("Model unavailable")

        docs = [{"content": "Some context", "score": 0.8, "id": "1"}]
        state = _make_state(relevant_docs=docs)
        result = await generate_node(state, llm_provider)

        assert "error" in result["answer"].lower()
        assert result["should_escalate"] is True
        assert "LLM generation failed" in result["escalation_reason"]

    @pytest.mark.asyncio
    async def test_generate_with_multiple_sources(self) -> None:
        """Should include all sources in response."""
        llm_provider = AsyncMock()
        llm_provider.generate.return_value = "Combined answer from sources 1 and 2."
        llm_provider.default_model = "llama3"

        docs = [
            {"content": "First document", "score": 0.9, "id": "1"},
            {"content": "Second document", "score": 0.7, "id": "2"},
        ]
        state = _make_state(relevant_docs=docs)
        result = await generate_node(state, llm_provider)

        assert len(result["sources"]) == 2


class TestEscalationNode:
    """Test suite for escalation_node."""

    @pytest.mark.asyncio
    async def test_escalation_sets_message(self) -> None:
        """Should set escalation message."""
        state = _make_state(should_escalate=True, escalation_reason="No relevant docs")
        result = await escalation_node(state)

        assert "human support agent" in result["answer"].lower()
        assert result["sources"] == []


class TestRunRAGPipeline:
    """Test suite for the full pipeline orchestrator."""

    @pytest.mark.asyncio
    async def test_pipeline_happy_path(self) -> None:
        """Full pipeline should retrieve, grade, and generate."""
        vector_store = AsyncMock()
        embedding_service = AsyncMock()
        llm_provider = AsyncMock()

        embedding_service.embed.return_value = [0.1, 0.2, 0.3]
        vector_store.search.return_value = [_make_search_result(score=0.9)]
        llm_provider.generate.return_value = "Here's your answer!"
        llm_provider.default_model = "test-model"

        result = await run_rag_pipeline(
            query="How do I reset my password?",
            tenant_id="tenant-123",
            vector_store=vector_store,
            embedding_service=embedding_service,
            llm_provider=llm_provider,
        )

        assert result["answer"] == "Here's your answer!"
        assert result["should_escalate"] is False
        assert len(result["sources"]) == 1

    @pytest.mark.asyncio
    async def test_pipeline_escalation_path(self) -> None:
        """Pipeline should escalate when no relevant docs found."""
        vector_store = AsyncMock()
        embedding_service = AsyncMock()
        llm_provider = AsyncMock()

        embedding_service.embed.return_value = [0.1, 0.2]
        vector_store.search.return_value = []

        result = await run_rag_pipeline(
            query="Something completely unknown?",
            tenant_id="tenant-123",
            vector_store=vector_store,
            embedding_service=embedding_service,
            llm_provider=llm_provider,
        )

        assert result["should_escalate"] is True
        assert "human support agent" in result["answer"].lower()
        llm_provider.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_pipeline_low_relevance_escalation(self) -> None:
        """Pipeline should escalate when docs exist but are low relevance."""
        vector_store = AsyncMock()
        embedding_service = AsyncMock()
        llm_provider = AsyncMock()

        embedding_service.embed.return_value = [0.1, 0.2]
        vector_store.search.return_value = [_make_search_result(score=0.1)]

        result = await run_rag_pipeline(
            query="Obscure question",
            tenant_id="tenant-123",
            vector_store=vector_store,
            embedding_service=embedding_service,
            llm_provider=llm_provider,
            relevance_threshold=0.5,
        )

        assert result["should_escalate"] is True
        llm_provider.generate.assert_not_called()
