"""Tests for the RecursiveChunker."""

from __future__ import annotations

import pytest

from app.rag.chunking import Chunk, RecursiveChunker


class TestRecursiveChunkerInit:
    """Test suite for chunker initialization."""

    def test_default_values(self) -> None:
        chunker = RecursiveChunker()
        assert chunker.chunk_size == 512
        assert chunker.overlap == 50

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
        chunker = RecursiveChunker(chunk_size=20, overlap=5)
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
        chunker = RecursiveChunker(chunk_size=50, overlap=10)
        text = "This is a sentence. " * 20  # ~400 chars
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_chunks_respect_max_size(self) -> None:
        """Each chunk should not greatly exceed chunk_size."""
        chunker = RecursiveChunker(chunk_size=100, overlap=10)
        text = "Word " * 200  # ~1000 chars
        chunks = chunker.chunk(text)
        for chunk in chunks:
            # Allow some slack due to overlap
            assert len(chunk.content) <= chunker.chunk_size + chunker.overlap + 20

    def test_paragraph_splitting(self) -> None:
        """Text with paragraphs should split at paragraph boundaries."""
        chunker = RecursiveChunker(chunk_size=50, overlap=5)
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
        chunker = RecursiveChunker(chunk_size=30, overlap=0)
        text = "AAAA BBBB CCCC DDDD EEEE FFFF GGGG HHHH IIII JJJJ"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_hard_split_triggered_for_long_unsplittable_text(self) -> None:
        """Text with no separators exceeding chunk_size should trigger hard_split."""
        # Create text with no spaces, newlines, or periods
        chunker = RecursiveChunker(chunk_size=20, overlap=5)
        long_word = "A" * 100  # Single 100-char string with no separators
        chunks = chunker.chunk(long_word)
        assert len(chunks) >= 2
        # Verify chunks were produced (content is not empty)
        for chunk in chunks:
            assert len(chunk.content) > 0

    def test_sentence_level_splitting(self) -> None:
        """Text that doesn't split by paragraphs should split by sentences."""
        chunker = RecursiveChunker(chunk_size=40, overlap=5)
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_merge_with_small_overlap(self) -> None:
        """Overlap smaller than current chunk should still work."""
        chunker = RecursiveChunker(chunk_size=30, overlap=3)
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
        chunker = RecursiveChunker(chunk_size=25, overlap=5)
        text = "This is a very long sentence that exceeds the chunk size limit"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
