"""BM25 keyword retrieval for hybrid search.

Builds a per-query BM25 index from tenant chunks and scores
the query against it. Complements vector search by catching
exact keyword matches that embedding models often miss
(product IDs, error codes, specific terms).

This module is stateless — no cached index. The corpus is
loaded fresh each query via ``VectorStore.get_all_documents()``.
For corpora under ~10K chunks this adds ~50ms of overhead.
"""

from __future__ import annotations

import structlog
from rank_bm25 import BM25Okapi

logger = structlog.get_logger(__name__)


def bm25_search(
    query: str,
    documents: list[dict[str, object]],
    k: int = 20,
) -> list[dict[str, object]]:
    """Score documents with BM25Okapi and return top-k ranked.

    Tokenization uses lowercased whitespace split — simple but
    effective for English support documentation. No external NLP
    dependency required.

    Args:
        query: User's search query.
        documents: List of dicts, each with at least ``content`` (str),
            ``metadata`` (dict), and ``id`` (str) keys.
        k: Maximum number of results to return.

    Returns:
        Top-k documents sorted by BM25 score (descending), each
        with an added ``score`` key normalized to [0, 1].
    """
    if not documents or not query.strip():
        return []

    # Build corpus — extract text and tokenize
    corpus_texts: list[str] = []
    for doc in documents:
        text = str(doc.get("content", ""))
        corpus_texts.append(text)

    tokenized_corpus = [text.lower().split() for text in corpus_texts]
    tokenized_query = query.lower().split()

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    raw_scores = bm25.get_scores(tokenized_query)

    # Normalize scores to [0, 1] via min-max scaling
    max_score = float(max(raw_scores)) if len(raw_scores) > 0 else 0.0
    min_score = float(min(raw_scores)) if len(raw_scores) > 0 else 0.0
    score_range = max_score - min_score

    scored_docs: list[tuple[float, int]] = []
    for idx, raw_score in enumerate(raw_scores):
        normalized = (float(raw_score) - min_score) / score_range if score_range > 0 else 0.0
        scored_docs.append((normalized, idx))

    # Sort by score descending, take top-k
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_k = scored_docs[:k]

    results: list[dict[str, object]] = []
    for score, idx in top_k:
        doc = documents[idx]
        results.append({
            "content": doc.get("content", ""),
            "metadata": doc.get("metadata", {}),
            "score": score,
            "id": doc.get("id", ""),
        })

    logger.debug(
        "bm25_search_complete",
        query=query[:100],
        corpus_size=len(documents),
        top_score=results[0]["score"] if results else 0.0,
        result_count=len(results),
    )

    return results
