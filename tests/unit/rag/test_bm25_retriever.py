"""Tests for BM25 keyword retrieval."""

from __future__ import annotations

from app.rag.bm25_retriever import bm25_search


def _make_docs(*texts: str) -> list[dict[str, object]]:
    """Create test documents from text strings."""
    return [
        {"content": text, "metadata": {"source": f"doc_{i}.txt"}, "id": f"doc-{i}"}
        for i, text in enumerate(texts)
    ]


class TestBM25Search:
    """Test suite for the bm25_search function."""

    def test_empty_corpus_returns_empty(self) -> None:
        """No documents should return empty results."""
        results = bm25_search("test query", [])
        assert results == []

    def test_empty_query_returns_empty(self) -> None:
        """Empty/whitespace query should return empty results."""
        docs = _make_docs("Some document text")
        assert bm25_search("", docs) == []
        assert bm25_search("   ", docs) == []

    def test_exact_keyword_match_ranks_highest(self) -> None:
        """Document with exact query keywords should rank higher."""
        docs = _make_docs(
            "Our shipping policy covers delivery within 5 days.",
            "Returns and exchanges can be processed within 30 days.",
            "P.O. box delivery is available for standard shipping only.",
        )
        results = bm25_search("P.O. box shipping", docs, k=3)
        assert len(results) == 3
        # The P.O. box document should be ranked first
        assert "P.O. box" in str(results[0]["content"])

    def test_returns_correct_dict_shape(self) -> None:
        """Each result should have content, metadata, score, id keys."""
        docs = _make_docs("Test document content")
        results = bm25_search("test", docs, k=1)
        assert len(results) == 1
        result = results[0]
        assert "content" in result
        assert "metadata" in result
        assert "score" in result
        assert "id" in result

    def test_scores_normalized_0_to_1(self) -> None:
        """BM25 scores should be min-max normalized to [0, 1]."""
        docs = _make_docs(
            "The quick brown fox jumps over the lazy dog.",
            "A completely different document about cooking recipes.",
            "The fox is quick and jumps high.",
        )
        results = bm25_search("quick fox jumps", docs, k=3)
        for result in results:
            score = float(result["score"])
            assert 0.0 <= score <= 1.0
        # Top result should be 1.0 (max after normalization)
        assert float(results[0]["score"]) == 1.0

    def test_k_limits_output(self) -> None:
        """Should return at most k results."""
        docs = _make_docs(*[f"Document number {i}" for i in range(10)])
        results = bm25_search("document number", docs, k=3)
        assert len(results) == 3

    def test_preserves_metadata_and_id(self) -> None:
        """Metadata and ID should pass through from input docs."""
        docs = [
            {"content": "Return policy details", "metadata": {"file": "policy.md"}, "id": "chunk-42"},
        ]
        results = bm25_search("return policy", docs, k=1)
        assert results[0]["metadata"] == {"file": "policy.md"}
        assert results[0]["id"] == "chunk-42"

    def test_no_matching_terms_still_returns_results(self) -> None:
        """BM25 scores everything, even with no keyword overlap."""
        docs = _make_docs("apple banana cherry")
        results = bm25_search("completely unrelated query", docs, k=1)
        # BM25 will still score the document (just very low)
        assert len(results) == 1

    def test_special_characters_handled(self) -> None:
        """Punctuation and special chars should not crash BM25."""
        docs = _make_docs(
            "Error code: ERR_CONN_RESET_4XX",
            "Status: 200 OK",
        )
        results = bm25_search("ERR_CONN_RESET_4XX", docs, k=2)
        assert len(results) == 2

    def test_single_document_corpus(self) -> None:
        """Single document corpus should work correctly."""
        docs = _make_docs("The only document in the collection.")
        results = bm25_search("document collection", docs, k=5)
        assert len(results) == 1
        # Single doc normalization: score should be 0.0 (range is 0)
        assert float(results[0]["score"]) == 0.0

    def test_case_insensitive_matching(self) -> None:
        """BM25 tokenization is lowercased, matching should be case-insensitive."""
        docs = _make_docs("PREMIUM Membership Benefits")
        results = bm25_search("premium membership", docs, k=1)
        assert len(results) == 1
