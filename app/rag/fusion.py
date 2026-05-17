"""Weighted Reciprocal Rank Fusion for hybrid retrieval.

Merges ranked lists from multiple retrieval methods (vector search,
BM25, etc.) using rank position rather than raw scores. This avoids
the need to normalize incompatible score distributions.

Formula:
    score(d) = Σ weight_i × 1/(k + rank_i(d))

Where k is a smoothing constant (default 60) and rank is the
1-based position in each list.
"""

from __future__ import annotations

from collections import defaultdict

import structlog

logger = structlog.get_logger(__name__)


def weighted_rrf(
    ranked_lists: list[list[dict[str, object]]],
    weights: list[float] | None = None,
    k: int = 60,
    top_n: int = 20,
) -> list[dict[str, object]]:
    """Fuse multiple ranked lists using Weighted Reciprocal Rank Fusion.

    Works with any number of input lists (1 for pure vector, 2 for
    hybrid, N for multi-model ensembles).

    Args:
        ranked_lists: Each sublist contains doc dicts ordered by
            relevance (best first). Each doc must have an ``id`` key.
        weights: Weight for each ranking list. ``None`` uses equal
            weights. Weights are relative (don't need to sum to 1.0).
        k: Smoothing constant. Higher values reduce the advantage
            of top-ranked items. Standard default is 60.
        top_n: Maximum number of results to return.

    Returns:
        Fused list of doc dicts sorted by combined RRF score
        (descending). Each doc retains its original ``score``
        (e.g., cosine similarity) and gains an ``rrf_score``
        field with the fusion rank score.

    Raises:
        ValueError: If weights length doesn't match ranked_lists length.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        msg = f"weights length ({len(weights)}) must match ranked_lists length ({len(ranked_lists)})"
        raise ValueError(msg)

    # Accumulate RRF scores by document ID
    rrf_scores: defaultdict[str, float] = defaultdict(float)
    # Keep track of the best doc dict for each ID (for metadata)
    doc_map: dict[str, dict[str, object]] = {}

    for list_idx, rank_list in enumerate(ranked_lists):
        weight = weights[list_idx]
        for rank, doc in enumerate(rank_list, start=1):
            doc_id = str(doc.get("id", f"unknown_{list_idx}_{rank}"))
            rrf_scores[doc_id] += weight * (1.0 / (k + rank))

            # Keep the doc with the highest original score
            if doc_id not in doc_map:
                doc_map[doc_id] = dict(doc)
            else:
                existing_score = float(doc_map[doc_id].get("score", 0))  # type: ignore[arg-type]
                new_score = float(doc.get("score", 0))  # type: ignore[arg-type]
                if new_score > existing_score:
                    doc_map[doc_id] = dict(doc)

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)

    results: list[dict[str, object]] = []
    for doc_id in sorted_ids[:top_n]:
        doc = doc_map[doc_id]
        # Preserve original score (e.g., cosine similarity) for downstream
        # grading; add RRF score as a separate field for observability.
        doc["rrf_score"] = rrf_scores[doc_id]
        results.append(doc)

    logger.debug(
        "rrf_fusion_complete",
        input_lists=len(ranked_lists),
        weights=weights,
        unique_docs=len(rrf_scores),
        output_count=len(results),
    )

    return results
