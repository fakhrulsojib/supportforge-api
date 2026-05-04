"""Vector store factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.infrastructure.vectorstore.chroma_adapter import ChromaAdapter

if TYPE_CHECKING:
    from app.config import Settings
    from app.domain.interfaces.vector_store import VectorStore


def get_vector_store(settings: Settings) -> VectorStore:
    """Create and return the configured vector store."""
    return ChromaAdapter(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection_prefix=settings.chroma_collection_prefix,
    )
