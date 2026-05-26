"""LangGraph RAG graph builder.

Builds a minimal StateGraph: retrieve → grade → END.
No tool nodes, no generate node, no stream_mode flag.

The graph handles ONLY retrieval and grading — the caller is
responsible for tool execution (``run_tool_loop``) and answer
generation (``generate_node`` or ``provider.stream``).

This ensures a single graph shape for both ``process_message``
and ``stream_message`` paths, eliminating dual-path bugs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from app.rag.pipeline import RAGState, grade_node, retrieve_node

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_rag_graph(
    vector_store: Any,
    embedding_service: Any,
    llm_provider: Any,
    *,
    relevance_threshold: float = 0.3,
    embedding_model: str | None = None,
    **kwargs: Any,
) -> CompiledStateGraph:
    """Build and compile the RAG StateGraph.

    Always: START → retrieve → grade → END

    Args:
        vector_store: VectorStore adapter instance.
        embedding_service: EmbeddingService instance.
        llm_provider: LLMProvider adapter instance (used by grade_node).
        relevance_threshold: Minimum similarity score for grading.
        embedding_model: Optional tenant-specific embedding model override.
        **kwargs: Additional keyword arguments (reserved for future use).

    Returns:
        Compiled LangGraph graph, ready for ``await graph.ainvoke(state)``.
    """

    async def _retrieve(state: RAGState) -> RAGState:
        """Retrieve documents via hybrid search."""
        return await retrieve_node(
            state,
            vector_store,
            embedding_service,
            embedding_model=embedding_model,
        )

    async def _grade(state: RAGState) -> RAGState:
        """Grade retrieved documents for relevance."""
        return await grade_node(
            state,
            llm_provider,
            relevance_threshold,
        )

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("grade", _grade)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("grade", END)

    return graph.compile()
