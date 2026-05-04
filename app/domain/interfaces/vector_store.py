"""Vector store interface (port).

Abstract base class defining the contract for vector database operations.
Concrete implementations (e.g., ChromaAdapter) live in infrastructure/.

NO framework imports allowed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single result from a vector similarity search."""

    content: str
    metadata: dict[str, object] = Field(default_factory=dict)
    score: float = 0.0
    id: str = ""


class VectorStore(ABC):
    """Port for vector database operations.

    All vector stores (ChromaDB, Pinecone, etc.) must implement this
    interface. Collections are namespaced by tenant_id for isolation.
    """

    @abstractmethod
    async def add_documents(
        self,
        tenant_id: str,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, object]],
        ids: list[str],
    ) -> None:
        """Add documents with embeddings to the tenant's collection.

        Args:
            tenant_id: Tenant identifier for collection namespacing.
            documents: Text content of each document chunk.
            embeddings: Pre-computed embedding vectors.
            metadatas: Metadata dicts for each document.
            ids: Unique identifiers for each document.
        """
        ...

    @abstractmethod
    async def search(
        self,
        tenant_id: str,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[SearchResult]:
        """Search for similar documents in the tenant's collection.

        Args:
            tenant_id: Tenant identifier.
            query_embedding: Query embedding vector.
            k: Number of results to return.

        Returns:
            List of SearchResults ordered by similarity (best first).
        """
        ...

    @abstractmethod
    async def delete_collection(self, tenant_id: str) -> None:
        """Delete an entire tenant's collection.

        Args:
            tenant_id: Tenant identifier.
        """
        ...

    @abstractmethod
    async def get_collection_stats(self, tenant_id: str) -> dict[str, object]:
        """Get statistics for a tenant's collection.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            Dict with keys like 'count', 'name', etc.
        """
        ...
