"""Tests for Weighted Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from app.rag.fusion import weighted_rrf


def _make_doc(doc_id: str, content: str = "", score: float = 0.0) -> dict[str, object]:
    """Create a test document dict."""
    return {"content": content, "metadata": {}, "score": score, "id": doc_id}


class TestWeightedRRF:
    """Test suite for the weighted_rrf function."""

    def test_empty_input_returns_empty(self) -> None:
        """No ranked lists should return empty results."""
        assert weighted_rrf([]) == []

    def test_single_list_returns_ranked(self) -> None:
        """Single list input should work (pure vector mode)."""
        docs = [_make_doc("a", score=0.9), _make_doc("b", score=0.7)]
        results = weighted_rrf([docs])
        assert len(results) == 2
        # First doc should still be first (rank 1 in the only list)
        assert results[0]["id"] == "a"

    def test_doc_in_both_lists_gets_boosted(self) -> None:
        """Document appearing in both lists should score higher."""
        vector = [_make_doc("shared"), _make_doc("vector_only")]
        bm25 = [_make_doc("shared"), _make_doc("bm25_only")]

        results = weighted_rrf([vector, bm25], weights=[1.0, 1.0])

        # "shared" appears in both → boosted to top
        assert results[0]["id"] == "shared"
        # The shared doc's RRF score should be higher than the others
        shared_score = float(results[0]["rrf_score"])
        others = [float(r["rrf_score"]) for r in results[1:]]
        assert all(shared_score > s for s in others)

    def test_weighted_favors_higher_weight(self) -> None:
        """Higher weight should favor that list's top results."""
        vector = [_make_doc("vec_top"), _make_doc("vec_second")]
        bm25 = [_make_doc("bm25_top"), _make_doc("bm25_second")]

        # Heavily favor BM25
        results = weighted_rrf([vector, bm25], weights=[0.1, 0.9])

        # BM25 top should outrank vector top due to higher weight
        assert results[0]["id"] == "bm25_top"

    def test_equal_weights_symmetric(self) -> None:
        """Equal weights should treat both lists symmetrically."""
        vector = [_make_doc("a"), _make_doc("b")]
        bm25 = [_make_doc("c"), _make_doc("d")]

        results = weighted_rrf([vector, bm25], weights=[1.0, 1.0])

        # a and c are both rank 1 in their lists — should have equal scores
        a_score = next(float(r["rrf_score"]) for r in results if r["id"] == "a")
        c_score = next(float(r["rrf_score"]) for r in results if r["id"] == "c")
        assert abs(a_score - c_score) < 1e-10

    def test_top_n_limits_output(self) -> None:
        """Should return at most top_n results."""
        docs = [_make_doc(f"doc-{i}") for i in range(10)]
        results = weighted_rrf([docs], top_n=3)
        assert len(results) == 3

    def test_default_equal_weights(self) -> None:
        """None weights should default to equal weights."""
        vector = [_make_doc("a")]
        bm25 = [_make_doc("b")]
        results = weighted_rrf([vector, bm25], weights=None)
        assert len(results) == 2

    def test_mismatched_weights_raises(self) -> None:
        """Weights count must match lists count."""
        with pytest.raises(ValueError, match="weights length"):
            weighted_rrf([[_make_doc("a")]], weights=[1.0, 2.0])

    def test_rrf_k_parameter_affects_scoring(self) -> None:
        """Smaller k gives more advantage to top-ranked documents."""
        docs = [_make_doc("first"), _make_doc("second")]

        # Small k: rank 1 gets much more credit
        results_small_k = weighted_rrf([docs], k=1)
        # Large k: rank difference is minimal
        results_large_k = weighted_rrf([docs], k=1000)

        score_diff_small = float(results_small_k[0]["rrf_score"]) - float(results_small_k[1]["rrf_score"])
        score_diff_large = float(results_large_k[0]["rrf_score"]) - float(results_large_k[1]["rrf_score"])

        # Score gap should be larger with smaller k
        assert score_diff_small > score_diff_large

    def test_preserves_best_metadata(self) -> None:
        """When a doc appears in multiple lists, keep the highest-scored version."""
        vector = [{"content": "text_v", "metadata": {"src": "vector"}, "score": 0.9, "id": "shared"}]
        bm25 = [{"content": "text_b", "metadata": {"src": "bm25"}, "score": 0.5, "id": "shared"}]

        results = weighted_rrf([vector, bm25], weights=[1.0, 1.0])
        assert len(results) == 1
        # Should keep the vector version (higher original score)
        assert results[0]["metadata"] == {"src": "vector"}

    def test_three_way_fusion(self) -> None:
        """Should work with 3+ ranked lists (future: multi-model ensembles)."""
        list1 = [_make_doc("a"), _make_doc("b")]
        list2 = [_make_doc("b"), _make_doc("c")]
        list3 = [_make_doc("c"), _make_doc("a")]

        results = weighted_rrf([list1, list2, list3], weights=[1.0, 1.0, 1.0])
        # All three docs appear in 2 lists each — all should be present
        ids = [r["id"] for r in results]
        assert set(ids) == {"a", "b", "c"}
