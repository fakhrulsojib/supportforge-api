"""Tests for the reranker interface, NoOp adapter, and factory."""

from __future__ import annotations

from unittest.mock import patch

from app.domain.interfaces.reranker import Reranker
from app.infrastructure.reranker.noop_reranker import NoOpReranker


def _make_doc(doc_id: str, content: str = "text") -> dict[str, object]:
    """Create a test document dict."""
    return {"content": content, "metadata": {}, "score": 0.5, "id": doc_id}


class TestNoOpReranker:
    """Test suite for the NoOpReranker adapter."""

    def test_implements_reranker_interface(self) -> None:
        """NoOpReranker should be a valid Reranker implementation."""
        reranker = NoOpReranker()
        assert isinstance(reranker, Reranker)

    def test_returns_docs_unchanged(self) -> None:
        """Should return documents in the same order."""
        reranker = NoOpReranker()
        docs = [_make_doc("a"), _make_doc("b"), _make_doc("c")]
        results = reranker.rerank("query", docs, top_k=3)
        assert [r["id"] for r in results] == ["a", "b", "c"]

    def test_truncates_to_top_k(self) -> None:
        """Should limit results to top_k."""
        reranker = NoOpReranker()
        docs = [_make_doc(f"doc-{i}") for i in range(10)]
        results = reranker.rerank("query", docs, top_k=3)
        assert len(results) == 3

    def test_empty_docs_returns_empty(self) -> None:
        """Empty input should return empty output."""
        reranker = NoOpReranker()
        assert reranker.rerank("query", []) == []

    def test_top_k_larger_than_docs(self) -> None:
        """top_k larger than doc count should return all docs."""
        reranker = NoOpReranker()
        docs = [_make_doc("a"), _make_doc("b")]
        results = reranker.rerank("query", docs, top_k=10)
        assert len(results) == 2


class TestRerankerFactory:
    """Test suite for the reranker factory."""

    def test_disabled_returns_noop(self) -> None:
        """Factory should return NoOp when reranker_enabled=False."""
        with patch("app.config.get_settings") as mock_settings:
            mock_settings.return_value.reranker_enabled = False
            mock_settings.return_value.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

            from app.infrastructure.reranker.factory import get_reranker

            reranker = get_reranker()
            assert isinstance(reranker, NoOpReranker)

    def test_enabled_but_missing_library_returns_noop(self) -> None:
        """Factory should fallback to NoOp if sentence-transformers not installed."""
        with patch("app.config.get_settings") as mock_settings:
            mock_settings.return_value.reranker_enabled = True
            mock_settings.return_value.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

            # sentence-transformers is not installed in this env,
            # so CrossEncoderReranker.__init__ will raise ImportError
            # and the factory should catch it and return NoOp
            from app.infrastructure.reranker.factory import get_reranker

            reranker = get_reranker()
            assert isinstance(reranker, NoOpReranker)
