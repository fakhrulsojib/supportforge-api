"""ChromaDB vector store adapter.

Implements the VectorStore port with ChromaDB as the backing store.
Collections are namespaced as ``{prefix}{tenant_id}`` for tenant isolation.
"""

from __future__ import annotations

import chromadb
import structlog

from app.domain.interfaces.vector_store import SearchResult, VectorStore

logger = structlog.get_logger(__name__)


class ChromaAdapter(VectorStore):
    """Concrete vector store backed by ChromaDB.

    Attributes:
        _client: ChromaDB client instance.
        _collection_prefix: Prefix for tenant-namespaced collections.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8001,
        collection_prefix: str = "tenant_",
    ) -> None:
        self._client = chromadb.HttpClient(host=host, port=port)
        self._collection_prefix = collection_prefix

    def _collection_name(self, tenant_id: str) -> str:
        """Build the namespaced collection name."""
        return f"{self._collection_prefix}{tenant_id}"

    def _get_or_create_collection(self, tenant_id: str) -> chromadb.Collection:
        """Get or create a collection for the tenant."""
        name = self._collection_name(tenant_id)
        return self._client.get_or_create_collection(name=name)

    async def add_documents(
        self,
        tenant_id: str,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, object]],
        ids: list[str],
    ) -> None:
        """Add documents with embeddings to the tenant's ChromaDB collection."""
        collection = self._get_or_create_collection(tenant_id)
        collection.add(
            documents=documents,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=metadatas,  # type: ignore[arg-type]
            ids=ids,
        )
        logger.info(
            "documents_added_to_chroma",
            tenant_id=tenant_id,
            count=len(documents),
        )

    async def search(
        self,
        tenant_id: str,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[SearchResult]:
        """Search for similar documents in ChromaDB."""
        collection = self._get_or_create_collection(tenant_id)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []
        if results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(documents)
            ids = results["ids"][0] if results["ids"] else [""] * len(documents)

            for doc, meta, dist, doc_id in zip(documents, metadatas, distances, ids, strict=False):
                # ChromaDB returns distances (lower = more similar)
                # Convert to similarity score (higher = more similar)
                similarity = 1.0 / (1.0 + dist)
                search_results.append(
                    SearchResult(
                        content=doc or "",
                        metadata=meta or {},
                        score=similarity,
                        id=doc_id,
                    )
                )

        return search_results

    async def delete_collection(self, tenant_id: str) -> None:
        """Delete a tenant's ChromaDB collection."""
        name = self._collection_name(tenant_id)
        try:
            self._client.delete_collection(name=name)
            logger.info("collection_deleted", tenant_id=tenant_id, collection=name)
        except Exception as e:
            logger.warning("collection_delete_failed", tenant_id=tenant_id, error=str(e))

    async def get_collection_stats(self, tenant_id: str) -> dict[str, object]:
        """Get statistics for a tenant's ChromaDB collection."""
        collection = self._get_or_create_collection(tenant_id)
        count = collection.count()
        return {
            "name": self._collection_name(tenant_id),
            "count": count,
            "tenant_id": tenant_id,
        }
