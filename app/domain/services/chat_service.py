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


def _group_sources_by_document(
    relevant_docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group retrieved chunks by source document.

    Multiple chunks from the same document are collapsed into one
    source entry with the highest relevance score among them.

    Args:
        relevant_docs: List of retrieved document chunks with metadata.

    Returns:
        De-duplicated sources keyed by filename, highest score wins.
    """
    doc_map: dict[str, dict[str, Any]] = {}

    for doc in relevant_docs:
        meta = doc.get("metadata", {})
        filename = meta.get("filename", "")
        document_id = meta.get("document_id", "")
        key = document_id or filename or doc.get("id", str(uuid.uuid4()))

        score = doc.get("score", 0)

        if key not in doc_map or score > doc_map[key]["score"]:
            doc_map[key] = {
                "filename": filename or "Unknown source",
                "document_id": document_id,
                "score": score,
                "id": key,
            }

    return list(doc_map.values())


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

        # Group sources by document (de-duplicate chunks from same file)
        grouped_sources = _group_sources_by_document(
            result.get("relevant_docs", [])
        )

        return {
            "answer": result.get("answer", ""),
            "conversation_id": conversation_id,
            "sources": grouped_sources,
            "escalated": result.get("should_escalate", False),
            "escalation_reason": result.get("escalation_reason", ""),
            "model_used": result.get("model_used", ""),
        }

    async def stream_message(
        self,
        message: str,
        tenant_id: str,
        user_id: str = "",
        conversation_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream a chat response token-by-token via the RAG pipeline.

        Runs retrieval and grading synchronously, then streams the
        LLM generation step as individual token frames. Persists the
        conversation and messages to the database on completion.

        Frame types:
            - ``{"type": "token", "data": "partial text"}``
            - ``{"type": "source", "data": {"filename": ..., "score": ..., ...}}``
            - ``{"type": "done", "data": {"conversation_id": ..., ...}}``

        Args:
            message: User's message text.
            tenant_id: Tenant context.
            user_id: Authenticated user's ID (for conversation persistence).
            conversation_id: Optional existing conversation ID.

        Yields:
            Structured frame dicts for WebSocket delivery.
        """
        is_new_conversation = not conversation_id
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
            answer_text = state.get("answer", "")
            yield {
                "type": "token",
                "data": answer_text,
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

            # Persist escalation to DB
            await self._persist_exchange(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                user_id=user_id,
                user_message=message,
                assistant_message=answer_text,
                sources=[],
                model_used="",
                is_new=is_new_conversation,
            )
            return

        # Step 3: Build context and stream generation
        relevant_docs = state.get("relevant_docs", [])
        context_parts: list[str] = []

        for doc in relevant_docs:
            # Label chunks with their actual document filename, not numbered sources
            filename = doc.get("metadata", {}).get("filename", "Document")
            context_parts.append(f"[From: {filename}]\n{doc['content']}")

        context = "\n\n---\n\n".join(context_parts)

        # Group sources by document for the UI
        grouped_sources = _group_sources_by_document(relevant_docs)

        system_prompt = (
            "You are a knowledgeable and professional customer support assistant. "
            "Your role is to help users by answering their questions accurately "
            "based on the company's internal documentation.\n\n"
            "## Rules\n"
            "1. Answer ONLY using the provided context below. "
            "Do NOT use any outside knowledge or make assumptions.\n"
            "2. If the context does not contain enough information to fully answer "
            "the question, clearly state what you do know and what you cannot confirm.\n"
            "3. Do NOT reference internal labels like 'Source 1' or 'Source 3' in your response. "
            "Instead, refer to information naturally (e.g., 'According to the benefits documentation...').\n"
            "4. Keep your answers concise, well-structured, and easy to read. "
            "Use bullet points or numbered lists for multiple items.\n"
            "5. Be friendly, professional, and empathetic in tone.\n"
            "6. If the question is completely unrelated to the provided context, "
            "politely let the user know you can only help with topics covered in the documentation.\n"
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"### Context (from company documentation):\n\n"
                    f"{context}\n\n"
                    f"---\n\n"
                    f"### Question:\n{message}"
                ),
            },
        ]

        # Yield grouped source citations before streaming tokens
        for source in grouped_sources:
            yield {"type": "source", "data": source}

        # Stream LLM tokens and accumulate full answer
        full_answer_parts: list[str] = []
        async for token in self._llm_provider.stream(messages=messages):  # type: ignore[attr-defined]
            full_answer_parts.append(token)
            yield {"type": "token", "data": token}

        model_used = getattr(self._llm_provider, "default_model", "")
        full_answer = "".join(full_answer_parts)

        # Done frame
        yield {
            "type": "done",
            "data": {
                "conversation_id": conversation_id,
                "model_used": model_used,
                "sources": grouped_sources,
                "escalated": False,
                "escalation_reason": "",
            },
        }

        # Persist to database
        await self._persist_exchange(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_id=user_id,
            user_message=message,
            assistant_message=full_answer,
            sources=grouped_sources,
            model_used=model_used,
            is_new=is_new_conversation,
        )

    async def _persist_exchange(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        user_id: str,
        user_message: str,
        assistant_message: str,
        sources: list[dict[str, Any]],
        model_used: str,
        is_new: bool,
    ) -> None:
        """Save the conversation and its messages to the database.

        Creates a new conversation record if ``is_new`` is True, then
        appends both the user message and the assistant response.

        This runs AFTER streaming is complete to avoid blocking the
        token-by-token delivery.

        Args:
            conversation_id: UUID of the conversation.
            tenant_id: Tenant context.
            user_id: User who initiated the chat.
            user_message: The user's original question.
            assistant_message: The full AI response.
            sources: Grouped source citations.
            model_used: LLM model name.
            is_new: Whether this is a brand-new conversation.
        """
        try:
            from app.domain.models.conversation import Message
            from app.domain.models.enums import MessageRole
            from app.infrastructure.database.connection import AsyncSessionLocal
            from app.infrastructure.database.repositories.conversation_repo import (
                SQLConversationRepository,
                SQLMessageRepository,
            )

            async with AsyncSessionLocal() as session:
                conv_repo = SQLConversationRepository(session)
                msg_repo = SQLMessageRepository(session)

                # Create conversation if new
                if is_new:
                    await conv_repo.create(
                        tenant_id=tenant_id,
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )

                # Save user message
                await msg_repo.create(
                    Message(
                        conversation_id=conversation_id,
                        role=MessageRole.USER,
                        content=user_message,
                    )
                )

                # Save assistant response (use fallback if LLM returned nothing)
                saved_content = assistant_message or "(No response generated)"
                await msg_repo.create(
                    Message(
                        conversation_id=conversation_id,
                        role=MessageRole.ASSISTANT,
                        content=saved_content,
                        sources_json=sources,
                        model_used=model_used,
                    )
                )

                await session.commit()

            logger.info(
                "chat_exchange_persisted",
                conversation_id=conversation_id,
                is_new=is_new,
            )
        except Exception:
            # Log but don't fail the chat — persistence is best-effort
            logger.error(
                "chat_persist_failed",
                conversation_id=conversation_id,
                exc_info=True,
            )
