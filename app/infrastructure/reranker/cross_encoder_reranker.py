"""Cross-encoder reranker adapter (sentence-transformers).

Uses a pre-trained cross-encoder model to jointly score
query–document pairs for fine-grained relevance ranking.
The model is loaded lazily on first use and cached for
subsequent queries.

Requires ``sentence-transformers`` (optional dependency).
Install via: ``pip install -e ".[reranker]"``
"""

from __future__ import annotations

import structlog

from app.domain.interfaces.reranker import Reranker

logger = structlog.get_logger(__name__)


class CrossEncoderReranker(Reranker):
    """Local cross-encoder reranker using sentence-transformers.

    Loads the model once at construction time (lazy — downloaded
    on first instantiation if not cached). Batch-scores all
    query–document pairs for efficiency.

    Attributes:
        _model: The loaded CrossEncoder model instance.
        _model_name: Name of the model for logging.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        from sentence_transformers import CrossEncoder

        self._model_name = model_name
        logger.info(
            "reranker_loading",
            model=model_name,
            note="First load downloads ~80MB model to ~/.cache/",
        )
        self._model = CrossEncoder(model_name)
        logger.info("reranker_loaded", model=model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict[str, object]],
        top_k: int = 5,
    ) -> list[dict[str, object]]:
        """Rerank documents using the cross-encoder.

        Constructs (query, document_content) pairs, scores them
        in batch, then returns the top-k by reranker score.

        Args:
            query: User's search query.
            documents: Candidate documents from fusion stage.
            top_k: Maximum results to return.

        Returns:
            Top-k documents sorted by cross-encoder relevance.
        """
        if not documents:
            return []

        # Build pairs for batch scoring
        pairs = [
            (query, str(doc.get("content", "")))
            for doc in documents
        ]

        # Batch prediction — returns numpy array of logit scores
        scores = self._model.predict(pairs, batch_size=32)

        # Attach scores and sort
        scored: list[tuple[float, int]] = [
            (float(score), idx) for idx, score in enumerate(scores)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, object]] = []
        for score, idx in scored[:top_k]:
            doc = dict(documents[idx])
            doc["reranker_score"] = score
            results.append(doc)

        logger.debug(
            "reranker_complete",
            model=self._model_name,
            input_count=len(documents),
            output_count=len(results),
            top_score=results[0]["reranker_score"] if results else 0.0,
        )

        return results
