"""Chat service — orchestrates the RAG pipeline for conversation processing.

Relocated from ``app.api.v1.chat_service`` to the domain services layer
to maintain hexagonal architecture (domain logic should not live in API layer).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import structlog

from app.rag.pipeline import run_rag_pipeline

if TYPE_CHECKING:
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
