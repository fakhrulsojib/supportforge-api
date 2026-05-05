"""Chat service — orchestrates the RAG pipeline for conversation processing.

Relocated from ``app.api.v1.chat_service`` to the domain services layer
to maintain hexagonal architecture (domain logic should not live in API layer).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import structlog

from app.rag.pipeline import (
    RAGState,
    escalation_node,
    grade_node,
    retrieve_node,
    run_rag_pipeline,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from app.domain.interfaces.llm_provider import LLMProvider
    from app.domain.interfaces.vector_store import VectorStore
    from app.rag.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)


class ChatService:
    """Orchestrates chat interactions through the RAG pipeline.

    This service bridges the API layer with the RAG pipeline,
    managing conversation state and coordinating all dependencies.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
    ) -> None:
        self._llm_provider = llm_provider
        self._vector_store = vector_store
        self._embedding_service = embedding_service

    async def process_message(
        self,
        message: str,
        tenant_id: str,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a user message through the RAG pipeline.

        Args:
            message: User's message text.
            tenant_id: Tenant context.
            conversation_id: Optional existing conversation ID.

        Returns:
            Dict with answer, sources, escalation status, etc.
        """
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        logger.info(
            "chat_process_message",
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            message_length=len(message),
        )

        # Run the RAG pipeline
        result = await run_rag_pipeline(
            query=message,
            tenant_id=tenant_id,
            vector_store=self._vector_store,
            embedding_service=self._embedding_service,
            llm_provider=self._llm_provider,
        )

        return {
            "answer": result.get("answer", ""),
            "conversation_id": conversation_id,
            "sources": result.get("sources", []),
            "escalated": result.get("should_escalate", False),
            "escalation_reason": result.get("escalation_reason", ""),
            "model_used": result.get("model_used", ""),
        }

    async def stream_message(
        self,
        message: str,
        tenant_id: str,
        conversation_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream a chat response token-by-token via the RAG pipeline.

        Runs retrieval and grading synchronously, then streams the
        LLM generation step as individual token frames. Yields
        structured dicts suitable for WebSocket JSON frames.

        Frame types:
            - ``{"type": "token", "data": "partial text"}``
            - ``{"type": "source", "data": {"content": ..., "score": ..., "id": ...}}``
            - ``{"type": "done", "data": {"conversation_id": ..., "model_used": ..., "sources": ...}}``

        Args:
            message: User's message text.
            tenant_id: Tenant context.
            conversation_id: Optional existing conversation ID.

        Yields:
            Structured frame dicts for WebSocket delivery.
        """
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        logger.info(
            "chat_stream_message",
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            message_length=len(message),
        )

        # Step 1: Retrieve + Grade (non-streaming)
        state: RAGState = {
            "query": message,
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

        state = await retrieve_node(state, self._vector_store, self._embedding_service)
        state = await grade_node(state, self._llm_provider)

        # Step 2: Check escalation
        if state.get("should_escalate"):
            state = await escalation_node(state)
            yield {
                "type": "token",
                "data": state.get("answer", ""),
            }
            yield {
                "type": "done",
                "data": {
                    "conversation_id": conversation_id,
                    "model_used": "",
                    "sources": [],
                    "escalated": True,
                    "escalation_reason": state.get("escalation_reason", ""),
                },
            }
            return

        # Step 3: Build context and stream generation
        relevant_docs = state.get("relevant_docs", [])
        sources: list[dict[str, Any]] = []
        context_parts: list[str] = []

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
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {message}"},
        ]

        # Yield source citations before streaming tokens
        for source in sources:
            yield {"type": "source", "data": source}

        # Stream LLM tokens
        async for token in self._llm_provider.stream(messages=messages):  # type: ignore[attr-defined]
            yield {"type": "token", "data": token}

        # Done frame
        yield {
            "type": "done",
            "data": {
                "conversation_id": conversation_id,
                "model_used": getattr(self._llm_provider, "default_model", ""),
                "sources": sources,
                "escalated": False,
                "escalation_reason": "",
            },
        }
