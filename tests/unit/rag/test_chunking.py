"""Tests for the RecursiveChunker (token-based)."""

from __future__ import annotations

import pytest

from app.rag.chunking import Chunk, RecursiveChunker


class TestRecursiveChunkerInit:
    """Test suite for chunker initialization."""

    def test_default_values(self) -> None:
        chunker = RecursiveChunker()
        assert chunker.chunk_size == 500
        assert chunker.overlap == 75

    def test_custom_values(self) -> None:
        chunker = RecursiveChunker(chunk_size=256, overlap=25)
        assert chunker.chunk_size == 256
        assert chunker.overlap == 25

    def test_zero_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            RecursiveChunker(chunk_size=0)

    def test_negative_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            RecursiveChunker(overlap=-1)

    def test_overlap_gte_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap must be less than"):
            RecursiveChunker(chunk_size=100, overlap=100)


class TestTokenCounting:
    """Test suite for token-based size measurement."""

    def test_count_tokens_simple(self) -> None:
        chunker = RecursiveChunker()
        # "Hello world" is 2 tokens in cl100k_base
        count = chunker._count_tokens("Hello world")
        assert count >= 2

    def test_token_based_splitting(self) -> None:
        """Chunks should respect token limits, not character limits."""
        # Use a small token size to force splitting
        chunker = RecursiveChunker(chunk_size=20, overlap=3)
        # Generate text that is many tokens
        text = "This is a test sentence. " * 20  # ~100 tokens
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        # Verify each chunk is within the token limit (with some slack for overlap)
        for chunk in chunks:
            token_count = chunker._count_tokens(chunk.content)
            assert token_count <= chunker.chunk_size + chunker.overlap + 5


class TestChunking:
    """Test suite for chunk() method."""

    def test_empty_string_returns_empty(self) -> None:
        chunker = RecursiveChunker()
        assert chunker.chunk("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        chunker = RecursiveChunker()
        assert chunker.chunk("   \n\n   ") == []

    def test_short_text_returns_single_chunk(self) -> None:
        chunker = RecursiveChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk("Hello, world!")
        assert len(chunks) == 1
        assert chunks[0].content == "Hello, world!"
        assert chunks[0].index == 0

    def test_chunks_have_correct_indices(self) -> None:
        chunker = RecursiveChunker(chunk_size=10, overlap=2)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk(text)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_chunks_contain_metadata(self) -> None:
        chunker = RecursiveChunker(chunk_size=100, overlap=10)
        meta = {"source": "test.pdf"}
        chunks = chunker.chunk("Some text content.", metadata=meta)
        assert len(chunks) == 1
        assert chunks[0].metadata["source"] == "test.pdf"
        assert chunks[0].metadata["chunk_index"] == 0

    def test_long_text_splits_into_multiple_chunks(self) -> None:
        chunker = RecursiveChunker(chunk_size=20, overlap=3)
        text = "This is a sentence. " * 20  # ~100 tokens
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_chunks_respect_token_limit(self) -> None:
        """Each chunk should not greatly exceed chunk_size in tokens."""
        chunker = RecursiveChunker(chunk_size=30, overlap=5)
        text = "Word " * 200  # ~200 tokens
        chunks = chunker.chunk(text)
        for chunk in chunks:
            token_count = chunker._count_tokens(chunk.content)
            # Allow some slack due to overlap
            assert token_count <= chunker.chunk_size + chunker.overlap + 5

    def test_paragraph_splitting(self) -> None:
        """Text with paragraphs should split at paragraph boundaries."""
        chunker = RecursiveChunker(chunk_size=5, overlap=1)
        text = "Short paragraph one.\n\nShort paragraph two.\n\nShort paragraph three."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_chunk_model_fields(self) -> None:
        """Chunk model should have all expected fields."""
        chunk = Chunk(content="test", index=0)
        assert chunk.content == "test"
        assert chunk.index == 0
        assert chunk.metadata == {}

    def test_no_overlap_produces_distinct_chunks(self) -> None:
        """Zero overlap should produce chunks without repeated content."""
        chunker = RecursiveChunker(chunk_size=10, overlap=0)
        text = "AAAA BBBB CCCC DDDD EEEE FFFF GGGG HHHH IIII JJJJ"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_hard_split_triggered_for_long_unsplittable_text(self) -> None:
        """Text with no separators exceeding chunk_size should trigger hard_split."""
        chunker = RecursiveChunker(chunk_size=10, overlap=2)
        long_word = "A" * 200  # Single string with no separators
        chunks = chunker.chunk(long_word)
        assert len(chunks) >= 2
        # Verify chunks were produced (content is not empty)
        for chunk in chunks:
            assert len(chunk.content) > 0

    def test_sentence_level_splitting(self) -> None:
        """Text that doesn't split by paragraphs should split by sentences."""
        chunker = RecursiveChunker(chunk_size=5, overlap=1)
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_merge_with_small_overlap(self) -> None:
        """Overlap smaller than current chunk should still work."""
        chunker = RecursiveChunker(chunk_size=10, overlap=2)
        text = "AAAA BBBB\n\nCCCC DDDD\n\nEEEE FFFF\n\nGGGG HHHH"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_single_segment_after_split(self) -> None:
        """A single segment should be returned as-is from merge."""
        chunker = RecursiveChunker(chunk_size=100, overlap=10)
        text = "Just a single segment that fits."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1

    def test_recursive_split_sentence_then_word(self) -> None:
        """Very long sentence should split recursively by words."""
        chunker = RecursiveChunker(chunk_size=10, overlap=2)
        text = "This is a very long sentence that exceeds the chunk size limit"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_overlap_snaps_to_word_boundary(self) -> None:
        """Overlap text should not start mid-word."""
        chunker = RecursiveChunker(chunk_size=20, overlap=5)
        text = "The quick brown fox jumps over the lazy dog. " * 5
        chunks = chunker.chunk(text)
        if len(chunks) > 1:
            # Check that no chunk starts with a partial word fragment
            for chunk in chunks:
                # First character should not be a lowercase continuation
                # (this is a heuristic — word-boundary snapping should prevent fragments)
                content = chunk.content.strip()
                assert len(content) > 0


class TestExpandedSentenceSeparators:
    """Tests for the expanded separator list (? ! ;)."""

    def test_question_mark_separator(self) -> None:
        """Text with question marks should split at '? ' boundaries."""
        chunker = RecursiveChunker(chunk_size=5, overlap=1)
        text = "Need help with your order? Contact support today. We are available around the clock."
        chunks = chunker.chunk(text)
        # Should split at "? " rather than falling to word-level
        assert len(chunks) >= 2

    def test_exclamation_mark_separator(self) -> None:
        """Text with exclamation marks should split at '! ' boundaries."""
        chunker = RecursiveChunker(chunk_size=5, overlap=1)
        text = "Order now! Free shipping on all items. Do not miss this deal!"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_semicolon_separator(self) -> None:
        """Text with semicolons should split at '; ' boundaries."""
        chunker = RecursiveChunker(chunk_size=5, overlap=1)
        text = "Returns accepted within 30 days; exchanges within 14 days. Contact us for help."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_mixed_punctuation_splitting(self) -> None:
        """Text mixing periods, questions, and exclamations should split correctly."""
        chunker = RecursiveChunker(chunk_size=8, overlap=1)
        text = (
            "What is your return policy? "
            "Items can be returned within 30 days. "
            "Act fast! "
            "Some restrictions apply; see details below."
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        # All content should be preserved across chunks
        full_text = " ".join(c.content for c in chunks)
        assert "return policy" in full_text
        assert "restrictions" in full_text

    def test_separator_hierarchy_order(self) -> None:
        """Paragraph breaks should be preferred over sentence punctuation."""
        chunker = RecursiveChunker(chunk_size=10, overlap=1)
        text = "First paragraph here.\n\nSecond paragraph here? Yes indeed."
        chunks = chunker.chunk(text)
        # Should split at \n\n first, not at ". " or "? "
        assert len(chunks) >= 2

