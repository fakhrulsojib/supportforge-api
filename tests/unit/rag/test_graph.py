"""Tests for the LangGraph RAG graph builder.

Verifies the graph structure (retrieve → grade → END) and its
integration with the existing pipeline nodes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from app.rag.graph import build_rag_graph
from app.rag.pipeline import RAGState


def _make_state(**overrides: Any) -> RAGState:
    """Create a default RAGState for testing."""
    defaults: RAGState = {
        "query": "What is your refund policy?",
        "tenant_id": "test-tenant",
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
    defaults.update(overrides)
    return defaults


class TestBuildRagGraph:
    """Test the graph builder function."""

    def test_graph_compiles_without_error(self) -> None:
        """build_rag_graph should return a compiled graph."""
        mock_vs = AsyncMock()
        mock_embed = AsyncMock()
        mock_llm = AsyncMock()

        compiled = build_rag_graph(mock_vs, mock_embed, mock_llm)

        # Compiled graph should be callable (has ainvoke)
        assert hasattr(compiled, "ainvoke"), "Compiled graph must have ainvoke method"

    def test_graph_has_correct_nodes(self) -> None:
        """Graph should contain exactly 'retrieve' and 'grade' nodes."""
        mock_vs = AsyncMock()
        mock_embed = AsyncMock()
        mock_llm = AsyncMock()

        compiled = build_rag_graph(mock_vs, mock_embed, mock_llm)
        graph = compiled.get_graph()

        node_ids = {n for n in graph.nodes if n not in ("__start__", "__end__")}
        assert node_ids == {"retrieve", "grade"}, (
            f"Expected exactly {{'retrieve', 'grade'}} nodes, got {node_ids}"
        )

    def test_graph_has_no_generate_node(self) -> None:
        """Graph should NOT contain generate, escalate, tool_decider, or tool_executor."""
        mock_vs = AsyncMock()
        mock_embed = AsyncMock()
        mock_llm = AsyncMock()

        compiled = build_rag_graph(mock_vs, mock_embed, mock_llm)
        graph = compiled.get_graph()

        node_ids = set(graph.nodes)
        forbidden = {"generate", "escalate", "tool_decider", "tool_executor"}
        overlap = node_ids & forbidden
        assert not overlap, f"Graph should not have these nodes: {overlap}"


class TestGraphInvoke:
    """Test the graph execution with mocked nodes."""

    @pytest.mark.asyncio
    async def test_happy_path_retrieve_and_grade(self) -> None:
        """Graph should call retrieve then grade and return graded state."""
        mock_vs = AsyncMock()
        mock_embed = AsyncMock()
        mock_llm = AsyncMock()

        # Mock retrieve_node to return some docs
        retrieved_docs = [
            {"content": "Refund policy: 30-day returns.", "score": 0.85, "id": "doc1"},
        ]

        async def fake_retrieve(state: RAGState, *args: Any, **kwargs: Any) -> RAGState:
            state["retrieved_docs"] = retrieved_docs
            return state

        async def fake_grade(state: RAGState, *args: Any, **kwargs: Any) -> RAGState:
            state["relevant_docs"] = state["retrieved_docs"]
            state["should_escalate"] = False
            return state

        with (
            patch("app.rag.graph.retrieve_node", side_effect=fake_retrieve),
            patch("app.rag.graph.grade_node", side_effect=fake_grade),
        ):
            compiled = build_rag_graph(mock_vs, mock_embed, mock_llm)
            initial = _make_state()
            result = await compiled.ainvoke(initial)

        assert result["relevant_docs"] == retrieved_docs
        assert result["should_escalate"] is False
        # Graph should NOT generate an answer
        assert result["answer"] == ""

    @pytest.mark.asyncio
    async def test_escalation_path_no_docs(self) -> None:
        """When no relevant docs, grade should set should_escalate=True."""
        mock_vs = AsyncMock()
        mock_embed = AsyncMock()
        mock_llm = AsyncMock()

        async def fake_retrieve(state: RAGState, *args: Any, **kwargs: Any) -> RAGState:
            state["retrieved_docs"] = []
            return state

        async def fake_grade(state: RAGState, *args: Any, **kwargs: Any) -> RAGState:
            state["relevant_docs"] = []
            state["should_escalate"] = True
            state["escalation_reason"] = "No documents found"
            return state

        with (
            patch("app.rag.graph.retrieve_node", side_effect=fake_retrieve),
            patch("app.rag.graph.grade_node", side_effect=fake_grade),
        ):
            compiled = build_rag_graph(mock_vs, mock_embed, mock_llm)
            initial = _make_state()
            result = await compiled.ainvoke(initial)

        assert result["should_escalate"] is True
        assert result["escalation_reason"] == "No documents found"
        # Graph should NOT generate an answer even on escalation
        assert result["answer"] == ""

    @pytest.mark.asyncio
    async def test_graph_returns_ragstate_shape(self) -> None:
        """Graph output should have all RAGState keys."""
        mock_vs = AsyncMock()
        mock_embed = AsyncMock()
        mock_llm = AsyncMock()

        async def fake_retrieve(state: RAGState, *args: Any, **kwargs: Any) -> RAGState:
            return state

        async def fake_grade(state: RAGState, *args: Any, **kwargs: Any) -> RAGState:
            return state

        with (
            patch("app.rag.graph.retrieve_node", side_effect=fake_retrieve),
            patch("app.rag.graph.grade_node", side_effect=fake_grade),
        ):
            compiled = build_rag_graph(mock_vs, mock_embed, mock_llm)
            initial = _make_state()
            result = await compiled.ainvoke(initial)

        expected_keys = {
            "query", "tenant_id", "retrieved_docs", "relevant_docs",
            "answer", "sources", "should_escalate", "escalation_reason",
            "model_used", "tokens_in", "tokens_out",
        }
        assert expected_keys.issubset(set(result.keys())), (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )

    @pytest.mark.asyncio
    async def test_embedding_model_passed_to_retrieve(self) -> None:
        """embedding_model kwarg should be forwarded to retrieve_node."""
        mock_vs = AsyncMock()
        mock_embed = AsyncMock()
        mock_llm = AsyncMock()
        captured_kwargs: dict[str, Any] = {}

        async def fake_retrieve(state: RAGState, *args: Any, **kwargs: Any) -> RAGState:
            captured_kwargs.update(kwargs)
            return state

        async def fake_grade(state: RAGState, *args: Any, **kwargs: Any) -> RAGState:
            return state

        with (
            patch("app.rag.graph.retrieve_node", side_effect=fake_retrieve),
            patch("app.rag.graph.grade_node", side_effect=fake_grade),
        ):
            compiled = build_rag_graph(
                mock_vs, mock_embed, mock_llm,
                embedding_model="custom-embed-model",
            )
            initial = _make_state()
            await compiled.ainvoke(initial)

        assert captured_kwargs.get("embedding_model") == "custom-embed-model"
