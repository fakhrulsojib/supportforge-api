"""Document chunking strategies for RAG pipeline.

Splits text into overlapping chunks suitable for embedding and
vector storage. Uses a recursive approach: split by paragraphs
first, then by sentences, then by words if needed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A chunk of text produced by the chunker."""

    content: str
    index: int
    metadata: dict[str, object] = Field(default_factory=dict)


class RecursiveChunker:
    """Splits text into overlapping chunks recursively.

    Strategy:
        1. Split by double newlines (paragraphs)
        2. If a paragraph exceeds chunk_size, split by sentences
        3. If a sentence exceeds chunk_size, split by words
        4. Merge small segments until chunk_size is reached
        5. Apply overlap between consecutive chunks

    Attributes:
        chunk_size: Target size in characters for each chunk.
        overlap: Number of overlapping characters between chunks.
    """

    def __init__(self, chunk_size: int = 800, overlap: int = 100) -> None:
        if chunk_size <= 0:
            msg = "chunk_size must be positive"
            raise ValueError(msg)
        if overlap < 0:
            msg = "overlap must be non-negative"
            raise ValueError(msg)
        if overlap >= chunk_size:
            msg = "overlap must be less than chunk_size"
            raise ValueError(msg)

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict[str, object] | None = None) -> list[Chunk]:
        """Split text into overlapping chunks.

        Args:
            text: Input text to chunk.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of Chunk objects with content and index.
        """
        if not text or not text.strip():
            return []

        base_metadata = metadata or {}

        # Split into segments that respect chunk_size
        segments = self._split_recursive(text)

        # Merge small segments and apply overlap
        chunks = self._merge_with_overlap(segments)

        return [
            Chunk(
                content=chunk_text,
                index=i,
                metadata={**base_metadata, "chunk_index": i},
            )
            for i, chunk_text in enumerate(chunks)
        ]

    def _split_recursive(self, text: str) -> list[str]:
        """Recursively split text into segments that fit within chunk_size."""
        separators = ["\n\n", "\n", ". ", " "]
        return self._split_with_separators(text, separators)

    def _split_with_separators(self, text: str, separators: list[str]) -> list[str]:
        """Split text using the first applicable separator."""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        if not separators:
            # Last resort: hard split at chunk_size
            return self._hard_split(text)

        separator = separators[0]
        parts = text.split(separator)

        segments: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{separator}{part}" if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    segments.append(current.strip())
                # If this single part is too long, recurse with next separator
                if len(part) > self.chunk_size:
                    sub_segments = self._split_with_separators(part, separators[1:])
                    segments.extend(sub_segments)
                else:
                    current = part
                    continue
                current = ""

        if current and current.strip():
            segments.append(current.strip())

        return segments

    def _hard_split(self, text: str) -> list[str]:
        """Hard split text at chunk_size boundaries."""
        segments = []
        for i in range(0, len(text), self.chunk_size):
            segment = text[i : i + self.chunk_size].strip()
            if segment:
                segments.append(segment)
        return segments

    def _merge_with_overlap(self, segments: list[str]) -> list[str]:
        """Merge segments and add overlap between consecutive chunks."""
        if not segments:
            return []

        if len(segments) == 1:
            return segments

        chunks: list[str] = []
        current = segments[0]

        for i in range(1, len(segments)):
            if len(current) + len(segments[i]) + 1 <= self.chunk_size:
                current = f"{current} {segments[i]}"
            else:
                chunks.append(current)
                # Add overlap from end of previous chunk
                if self.overlap > 0 and len(current) > self.overlap:
                    overlap_text = current[-self.overlap :]
                    current = f"{overlap_text} {segments[i]}"
                else:
                    current = segments[i]

        if current:
            chunks.append(current)

        return chunks
