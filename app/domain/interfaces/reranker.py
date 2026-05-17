"""Reranker interface (port).

Abstract base class defining the contract for document reranking.
Concrete implementations (e.g., CrossEncoderReranker, CohereReranker)
live in ``infrastructure/reranker/``.

Swap between local models, API-based rerankers, or no-op
implementations without changing any calling code.

NO framework imports allowed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Reranker(ABC):
    """Port for document reranking.

    All reranker implementations must implement this interface,
    enabling zero-change provider swaps via the adapter pattern.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[dict[str, object]],
        top_k: int = 5,
    ) -> list[dict[str, object]]:
        """Rerank documents by relevance to query.

        Args:
            query: The user's search query.
            documents: Candidate documents to rerank. Each dict has
                at least ``content``, ``metadata``, ``score``, ``id``.
            top_k: Maximum number of documents to return.

        Returns:
            Top-k documents re-ordered by reranker relevance score.
            Original ``score`` must be preserved (grade_node depends on it).
            Reranker confidence is stored in ``reranker_score``.
        """
        ...
