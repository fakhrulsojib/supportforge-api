"""Chat API router — POST /api/v1/chat."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Header

from app.api.v1.chat_service import ChatService
from app.api.v1.schemas import ChatRequest, ChatResponse, SourceCitation
from app.config import get_settings
from app.core.exceptions import TenantNotFoundError
from app.infrastructure.llm.factory import get_llm_provider
from app.infrastructure.vectorstore.chroma_adapter import ChromaAdapter
from app.rag.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


def _build_chat_service() -> ChatService:
    """Build the ChatService with all dependencies.

    In production, these would be injected via FastAPI Depends
    and managed in the lifespan. For Phase 1, we create them inline.
    """
    settings = get_settings()

    llm_provider = get_llm_provider(settings)
    vector_store = ChromaAdapter(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection_prefix=settings.chroma_collection_prefix,
    )
    embedding_service = EmbeddingService(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
        cf_client_id=settings.cf_ollama_id,
        cf_client_secret=settings.cf_ollama_secret,
    )

    return ChatService(
        llm_provider=llm_provider,
        vector_store=vector_store,
        embedding_service=embedding_service,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    x_tenant_id: str = Header(..., alias="X-Tenant-ID"),
) -> ChatResponse:
    """Process a chat message through the RAG pipeline.

    Requires the X-Tenant-ID header for multi-tenant isolation.

    Args:
        request: Chat request with message and optional conversation_id.
        x_tenant_id: Tenant identifier from header.

    Returns:
        ChatResponse with AI-generated answer and source citations.

    Raises:
        TenantNotFoundError: If X-Tenant-ID header is missing.
    """
    if not x_tenant_id:
        raise TenantNotFoundError()

    chat_service = _build_chat_service()

    result = await chat_service.process_message(
        message=request.message,
        tenant_id=x_tenant_id,
        conversation_id=request.conversation_id,
    )

    sources = [
        SourceCitation(
            content=s.get("content", ""),
            score=s.get("score", 0.0),
            id=s.get("id", ""),
        )
        for s in result.get("sources", [])
    ]

    return ChatResponse(
        answer=result["answer"],
        conversation_id=result["conversation_id"],
        sources=sources,
        escalated=result.get("escalated", False),
        escalation_reason=result.get("escalation_reason", ""),
        model_used=result.get("model_used", ""),
    )
