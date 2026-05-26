"""Tests for the RAG pipeline nodes and orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock

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
        vector_store.get_all_documents.return_value = [_make_search_result()]

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
        vector_store.get_all_documents.return_value = []

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

    @pytest.mark.asyncio
    async def test_retrieve_hybrid_includes_bm25_only_docs(self) -> None:
        """Doc found only by BM25 (not vector) should appear in results via fusion."""
        vector_store = AsyncMock()
        embedding_service = AsyncMock()
        embedding_service.embed.return_value = [0.1, 0.2, 0.3]

        # Vector finds doc-A only
        vector_store.search.return_value = [
            _make_search_result(content="Vector hit", score=0.8),
        ]
        # Corpus contains doc-A AND doc-B (BM25 can find doc-B by keyword)
        vector_store.get_all_documents.return_value = [
            SearchResult(content="Vector hit", metadata={}, score=0.0, id="doc-1"),
            SearchResult(
                content="P.O. box delivery keyword match",
                metadata={},
                score=0.0,
                id="doc-bm25-only",
            ),
        ]

        state = _make_state(query="P.O. box delivery")
        result = await retrieve_node(state, vector_store, embedding_service)

        # Both docs should appear (vector hit via vector, BM25 hit via keyword match)
        result_ids = [doc["id"] for doc in result["retrieved_docs"]]
        assert "doc-1" in result_ids
        assert "doc-bm25-only" in result_ids

    @pytest.mark.asyncio
    async def test_retrieve_bm25_failure_degrades_to_vector_only(self) -> None:
        """If BM25/get_all_documents fails, should degrade to vector-only results."""
        vector_store = AsyncMock()
        embedding_service = AsyncMock()
        embedding_service.embed.return_value = [0.1, 0.2, 0.3]

        vector_store.search.return_value = [_make_search_result(score=0.9)]
        # Simulate ChromaDB failure on get_all_documents
        vector_store.get_all_documents.side_effect = RuntimeError("ChromaDB connection lost")

        state = _make_state()
        result = await retrieve_node(state, vector_store, embedding_service)

        # Should still return vector results despite BM25 failure
        assert len(result["retrieved_docs"]) >= 1
        assert result["retrieved_docs"][0]["content"] == "Test content"

    @pytest.mark.asyncio
    async def test_retrieve_bm25_respects_document_limit(self) -> None:
        """Should limit retrieved docs to 1000 to prevent OOM."""
        vector_store = AsyncMock()
        embedding_service = AsyncMock()
        embedding_service.embed.return_value = [0.1, 0.2, 0.3]
        vector_store.search.return_value = [_make_search_result()]
        vector_store.get_all_documents.return_value = [_make_search_result()]

        state = _make_state()
        await retrieve_node(state, vector_store, embedding_service)

        vector_store.get_all_documents.assert_called_once_with(
            "tenant-123", limit=1000
        )



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

    @pytest.mark.asyncio
    async def test_generate_with_history_messages(self) -> None:
        """History messages should be included in the LLM call."""
        llm_provider = AsyncMock()
        llm_provider.generate.return_value = "Based on our previous conversation..."
        llm_provider.default_model = "test-model"

        docs = [{"content": "Return policy: 30 days.", "score": 0.9, "id": "1"}]
        history = [
            {"role": "user", "content": "What's your return policy?"},
            {"role": "assistant", "content": "You can return within 30 days."},
        ]
        state = _make_state(relevant_docs=docs)
        result = await generate_node(
            state, llm_provider, history_messages=history,
        )

        assert result["answer"] == "Based on our previous conversation..."
        # Verify the messages passed to LLM include history
        call_args = llm_provider.generate.call_args
        messages = call_args.kwargs["messages"]
        # system + 2 history + user = 4 messages
        assert len(messages) == 4
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What's your return policy?"
        assert messages[2]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_generate_with_agent_config(self) -> None:
        """Agent config should customize the system prompt."""
        llm_provider = AsyncMock()
        llm_provider.generate.return_value = "Custom bot response."
        llm_provider.default_model = "test-model"

        docs = [{"content": "Some doc.", "score": 0.9, "id": "1"}]
        config = {"custom_prompt": "You are PirateBot. Say arrr!"}
        state = _make_state(relevant_docs=docs)
        result = await generate_node(
            state, llm_provider, agent_config=config,
        )

        call_args = llm_provider.generate.call_args
        messages = call_args.kwargs["messages"]
        # System prompt should contain the custom prompt
        assert "PirateBot" in messages[0]["content"]
        assert "arrr" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_generate_with_history_and_config(self) -> None:
        """Both history and config should work together."""
        llm_provider = AsyncMock()
        llm_provider.generate.return_value = "Combined response."
        llm_provider.default_model = "test-model"

        docs = [{"content": "Doc content.", "score": 0.9, "id": "1"}]
        history = [{"role": "user", "content": "Previous Q"}]
        config = {"agent_name": "TestBot", "company_name": "TestCo"}
        state = _make_state(relevant_docs=docs)
        result = await generate_node(
            state, llm_provider,
            history_messages=history, agent_config=config,
        )

        call_args = llm_provider.generate.call_args
        messages = call_args.kwargs["messages"]
        # system + 1 history + user = 3 messages
        assert len(messages) == 3
        # System prompt should have tenant config
        assert "TestBot" in messages[0]["content"]
        assert "TestCo" in messages[0]["content"]


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
        """Pipeline should retrieve and grade (no answer generation)."""
        vector_store = AsyncMock()
        embedding_service = AsyncMock()
        llm_provider = AsyncMock()

        embedding_service.embed.return_value = [0.1, 0.2, 0.3]
        vector_store.search.return_value = [_make_search_result(score=0.9)]
        vector_store.get_all_documents.return_value = [_make_search_result(score=0.9)]
        llm_provider.default_model = "test-model"

        result = await run_rag_pipeline(
            query="How do I reset my password?",
            tenant_id="tenant-123",
            vector_store=vector_store,
            embedding_service=embedding_service,
            llm_provider=llm_provider,
        )

        # Pipeline now only retrieves + grades — no answer generated
        assert result["should_escalate"] is False
        assert len(result["relevant_docs"]) > 0
        assert result["answer"] == ""  # No generation — caller handles this
        # generate should NOT be called by the pipeline
        llm_provider.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_pipeline_escalation_path(self) -> None:
        """Pipeline should set escalation flag when no relevant docs found."""
        vector_store = AsyncMock()
        embedding_service = AsyncMock()
        llm_provider = AsyncMock()

        embedding_service.embed.return_value = [0.1, 0.2]
        vector_store.search.return_value = []
        vector_store.get_all_documents.return_value = []

        result = await run_rag_pipeline(
            query="Something completely unknown?",
            tenant_id="tenant-123",
            vector_store=vector_store,
            embedding_service=embedding_service,
            llm_provider=llm_provider,
        )

        assert result["should_escalate"] is True
        # Pipeline no longer generates escalation answer — caller handles
        assert result["answer"] == ""
        llm_provider.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_pipeline_low_relevance_escalation(self) -> None:
        """Pipeline should escalate when docs exist but are low relevance."""
        vector_store = AsyncMock()
        embedding_service = AsyncMock()
        llm_provider = AsyncMock()

        embedding_service.embed.return_value = [0.1, 0.2]
        vector_store.search.return_value = [_make_search_result(score=0.1)]
        vector_store.get_all_documents.return_value = [_make_search_result(score=0.1)]

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

