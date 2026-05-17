"""No-op reranker — passthrough when reranking is disabled.

Returns documents unchanged (truncated to top_k). Used when
``reranker_enabled=False`` in config or when the
sentence-transformers library is not installed.
"""

from __future__ import annotations

from app.domain.interfaces.reranker import Reranker


class NoOpReranker(Reranker):
    """Passthrough reranker that returns documents unchanged.

    Used as the default when reranking is disabled via config.
    Implements the Reranker interface so calling code never
    needs to check whether reranking is active.
    """

    def rerank(
        self,
        query: str,
        documents: list[dict[str, object]],
        top_k: int = 5,
    ) -> list[dict[str, object]]:
        """Return documents unchanged, truncated to top_k."""
        return documents[:top_k]
