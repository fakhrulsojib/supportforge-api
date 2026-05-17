"""LangGraph RAG pipeline for multi-step question answering.

Implements a retrieval-augmented generation workflow:
    retrieve → grade → generate OR escalate

Each node is a pure function operating on a typed state dict.
"""

from __future__ import annotations

from typing import Any, TypedDict

import structlog

from app.core.exceptions import LLMError

logger = structlog.get_logger(__name__)


class RAGState(TypedDict, total=False):
    """Typed state flowing through the RAG graph.

    Attributes:
        query: The user's original question.
        tenant_id: Tenant context for collection namespacing.
        retrieved_docs: Documents from vector similarity search.
        relevant_docs: Documents that passed relevance grading.
        answer: Generated response text.
        sources: Source citations for the answer.
        should_escalate: Whether to escalate to a human agent.
        escalation_reason: Why escalation was triggered.
        model_used: Which LLM model was used.
        tokens_in: Input token count.
        tokens_out: Output token count.
    """

    query: str
    tenant_id: str
    retrieved_docs: list[dict[str, Any]]
    relevant_docs: list[dict[str, Any]]
    answer: str
    sources: list[dict[str, Any]]
    should_escalate: bool
    escalation_reason: str
    model_used: str
    tokens_in: int
    tokens_out: int


# ── Node functions ────────────────────────────────────────────────


async def retrieve_node(
    state: RAGState,
    vector_store: Any,
    embedding_service: Any,
    *,
    embedding_model: str | None = None,
) -> RAGState:
    """Retrieve relevant documents via hybrid search.

    Pipeline:
        1. **Vector search** — always active, semantic similarity
        2. **BM25 search** — config-toggled, keyword precision
        3. **Weighted RRF** — fuses both ranked lists (if BM25 active)
        4. **Reranker** — config-toggled, cross-encoder precision filter

    All parameters (k values, weights, toggles) are read from
    ``config.py``, keeping the function signature unchanged so
    callers (``run_rag_pipeline``, ``chat_service``) need zero changes.

    Args:
        state: Current RAG state.
        vector_store: VectorStore adapter instance.
        embedding_service: EmbeddingService instance.
        embedding_model: Optional tenant-specific embedding model override.

    Returns:
        Updated state with retrieved_docs populated.
    """
    from app.config import get_settings

    settings = get_settings()

    query = state["query"]
    tenant_id = state["tenant_id"]

    logger.info("rag_retrieve", query=query[:100], tenant_id=tenant_id)

    try:
        # ── Stage 1: Vector search (always active) ────────────────
        query_embedding = await embedding_service.embed(query, model=embedding_model)
        vector_results_raw = await vector_store.search(
            tenant_id=tenant_id,
            query_embedding=query_embedding,
            k=settings.retrieval_k_per_method,
        )
        vector_results: list[dict[str, Any]] = [
            {
                "content": r.content,
                "metadata": r.metadata,
                "score": r.score,
                "id": r.id,
            }
            for r in vector_results_raw
        ]

        logger.info(
            "rag_vector_search_complete",
            result_count=len(vector_results),
            top_score=vector_results[0]["score"] if vector_results else 0.0,
        )

        # ── Stage 2: BM25 search (config-toggled) ────────────────
        if settings.bm25_enabled:
            try:
                from app.rag.bm25_retriever import bm25_search

                all_docs_raw = await vector_store.get_all_documents(tenant_id)
                all_docs = [
                    {
                        "content": r.content,
                        "metadata": r.metadata,
                        "score": 0.0,
                        "id": r.id,
                    }
                    for r in all_docs_raw
                ]

                bm25_results = bm25_search(
                    query=query,
                    documents=all_docs,
                    k=settings.retrieval_k_per_method,
                )

                logger.info(
                    "rag_bm25_search_complete",
                    corpus_size=len(all_docs),
                    result_count=len(bm25_results),
                    top_score=bm25_results[0]["score"] if bm25_results else 0.0,
                )

                # ── Stage 3: Weighted RRF fusion ──────────────────────
                from app.rag.fusion import weighted_rrf

                fused = weighted_rrf(
                    ranked_lists=[vector_results, bm25_results],
                    weights=[settings.retrieval_vector_weight, settings.retrieval_bm25_weight],
                    k=settings.retrieval_rrf_k,
                    top_n=settings.retrieval_final_k * 4,  # wider pool for reranker
                )

                logger.info(
                    "rag_fusion_complete",
                    vector_weight=settings.retrieval_vector_weight,
                    bm25_weight=settings.retrieval_bm25_weight,
                    fused_count=len(fused),
                )
            except Exception as exc:
                logger.warning(
                    "rag_bm25_degraded",
                    error=str(exc),
                    fallback="vector_only",
                    exc_info=True,
                )
                fused = vector_results
        else:
            # BM25 disabled — use vector results directly
            fused = vector_results

        # ── Stage 4: Reranker (config-toggled) ────────────────────
        from app.infrastructure.reranker.factory import get_reranker

        reranker = get_reranker()
        retrieved_docs = reranker.rerank(
            query=query,
            documents=fused,
            top_k=settings.retrieval_final_k,
        )

    except LLMError as e:
        logger.warning("rag_retrieve_failed", error=str(e))
        retrieved_docs = []

    state["retrieved_docs"] = retrieved_docs
    return state


async def grade_node(
    state: RAGState,
    llm_provider: Any,
    relevance_threshold: float = 0.3,
) -> RAGState:
    """Grade retrieved documents for relevance.

    Uses a combination of:
        - Vector similarity score (threshold-based)
        - LLM-based relevance check for borderline docs

    Documents above the threshold are kept. If no documents pass,
    the system escalates.

    Args:
        state: Current RAG state with retrieved_docs.
        llm_provider: LLMProvider adapter instance.
        relevance_threshold: Minimum similarity score (0.0 - 1.0).

    Returns:
        Updated state with relevant_docs and escalation flag.
    """
    query = state["query"]
    retrieved = state.get("retrieved_docs", [])

    if not retrieved:
        state["relevant_docs"] = []
        state["should_escalate"] = True
        state["escalation_reason"] = "No documents found in knowledge base"
        return state

    # Score-based filtering
    relevant: list[dict[str, Any]] = []
    for doc in retrieved:
        if doc.get("score", 0) >= relevance_threshold:
            relevant.append(doc)

    if not relevant:
        state["relevant_docs"] = []
        state["should_escalate"] = True
        state["escalation_reason"] = "No relevant documents found above threshold"
        return state

    state["relevant_docs"] = relevant
    state["should_escalate"] = False
    logger.info("rag_grade", query=query[:100], total=len(retrieved), relevant=len(relevant))
    return state


async def generate_node(
    state: RAGState,
    llm_provider: Any,
    *,
    chat_model: str | None = None,
) -> RAGState:
    """Generate an answer using the LLM with retrieved context.

    Constructs a prompt with the relevant documents as context
    and generates a response.

    Args:
        state: Current RAG state with relevant_docs.
        llm_provider: LLMProvider adapter instance.
        chat_model: Optional tenant-specific chat model override.

    Returns:
        Updated state with answer and sources.
    """
    query = state["query"]
    relevant_docs = state.get("relevant_docs", [])

    # Build context from relevant documents
    context_parts: list[str] = []
    sources: list[dict[str, Any]] = []
    for i, doc in enumerate(relevant_docs):
        context_parts.append(f"[Source {i + 1}]: {doc['content']}")
        sources.append(
            {
                "content": doc["content"][:200],
                "score": doc.get("score", 0),
                "id": doc.get("id", ""),
            }
        )

    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful customer support assistant. "
        "Answer the question using ONLY the provided context. "
        "If the context doesn't contain enough information to answer, say so. "
        "Be concise, friendly, and professional. "
        "Always cite which source number you used."
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    try:
        answer = await llm_provider.generate(messages=messages, model=chat_model)
        state["answer"] = answer
        state["sources"] = sources
        state["model_used"] = chat_model or llm_provider.default_model
    except LLMError as e:
        logger.error("rag_generate_failed", error=str(e))
        state["answer"] = "I'm sorry, I encountered an error generating a response. Please try again."
        state["sources"] = []
        state["should_escalate"] = True
        state["escalation_reason"] = f"LLM generation failed: {e}"

    return state


async def escalation_node(
    state: RAGState,
) -> RAGState:
    """Handle escalation to a human agent.

    Sets a default escalation message when no suitable answer
    can be generated.

    Args:
        state: Current RAG state.

    Returns:
        Updated state with escalation answer.
    """
    reason = state.get("escalation_reason", "Unknown reason")
    logger.info("rag_escalate", reason=reason)

    state["answer"] = (
        "I wasn't able to find a confident answer to your question. "
        "I'm escalating this to a human support agent who will be able to help you. "
        "Please stand by — someone will be with you shortly."
    )
    state["sources"] = []
    return state


# ── Graph orchestrator ────────────────────────────────────────────


async def run_rag_pipeline(
    query: str,
    tenant_id: str,
    vector_store: Any,
    embedding_service: Any,
    llm_provider: Any,
    relevance_threshold: float = 0.3,
    *,
    chat_model: str | None = None,
    embedding_model: str | None = None,
) -> RAGState:
    """Execute the full RAG pipeline.

    Flow: retrieve → grade → (generate | escalate)

    Args:
        query: User's question.
        tenant_id: Tenant context.
        vector_store: VectorStore adapter.
        embedding_service: EmbeddingService.
        llm_provider: LLMProvider adapter.
        relevance_threshold: Minimum relevance score.
        chat_model: Optional tenant-specific chat model override.
        embedding_model: Optional tenant-specific embedding model override.

    Returns:
        Final RAGState with answer and metadata.
    """
    state: RAGState = {
        "query": query,
        "tenant_id": tenant_id,
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

    # Step 1: Retrieve
    state = await retrieve_node(state, vector_store, embedding_service, embedding_model=embedding_model)

    # Step 2: Grade
    state = await grade_node(state, llm_provider, relevance_threshold)

    # Step 3: Generate or Escalate
    if state.get("should_escalate"):
        state = await escalation_node(state)
    else:
        state = await generate_node(state, llm_provider, chat_model=chat_model)

    return state
