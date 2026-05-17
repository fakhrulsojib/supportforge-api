"""Document chunking strategies for RAG pipeline.

Splits text into overlapping chunks suitable for embedding and
vector storage. Uses a recursive approach: split by paragraphs
first, then by sentences, then by words if needed.

All size measurements use **token counts** (via tiktoken) rather
than raw character counts, ensuring alignment with embedding
model context limits.
"""

from __future__ import annotations

import tiktoken
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

    All size comparisons are **token-based** using tiktoken.

    Attributes:
        chunk_size: Target size in **tokens** for each chunk.
        overlap: Number of overlapping **tokens** between chunks.
        encoding: tiktoken encoding used for token counting.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 75,
        encoding_name: str = "cl100k_base",
    ) -> None:
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
        self._encoding = tiktoken.get_encoding(encoding_name)

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in text."""
        return len(self._encoding.encode(text))

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

        # Split into segments that respect chunk_size (in tokens)
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
        """Recursively split text into segments that fit within chunk_size tokens."""
        separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", " "]
        return self._split_with_separators(text, separators)

    def _split_with_separators(self, text: str, separators: list[str]) -> list[str]:
        """Split text using the first applicable separator."""
        if self._count_tokens(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        if not separators:
            # Last resort: hard split at chunk_size tokens
            return self._hard_split(text)

        separator = separators[0]
        parts = text.split(separator)

        segments: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{separator}{part}" if current else part
            if self._count_tokens(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    segments.append(current.strip())
                # If this single part is too long, recurse with next separator
                if self._count_tokens(part) > self.chunk_size:
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
        """Hard split text at chunk_size token boundaries.

        Uses the tokenizer to split precisely at token boundaries,
        ensuring no partial tokens and no mid-word cuts.
        """
        tokens = self._encoding.encode(text)
        segments = []
        for i in range(0, len(tokens), self.chunk_size):
            token_slice = tokens[i : i + self.chunk_size]
            segment = self._encoding.decode(token_slice).strip()
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
            if self._count_tokens(f"{current} {segments[i]}") <= self.chunk_size:
                current = f"{current} {segments[i]}"
            else:
                chunks.append(current)
                # Add overlap from end of previous chunk (token-aware)
                if self.overlap > 0:
                    overlap_text = self._get_token_overlap(current)
                    current = f"{overlap_text} {segments[i]}"
                else:
                    current = segments[i]

        if current:
            chunks.append(current)

        return chunks

    def _get_token_overlap(self, text: str) -> str:
        """Extract the last `self.overlap` tokens from text, snapped to a word boundary."""
        tokens = self._encoding.encode(text)
        if len(tokens) <= self.overlap:
            return text

        overlap_tokens = tokens[-self.overlap :]
        overlap_text = self._encoding.decode(overlap_tokens)

        # Snap to nearest word boundary to avoid mid-word cuts
        space_idx = overlap_text.find(" ")
        if space_idx > 0:
            overlap_text = overlap_text[space_idx + 1 :]

        return overlap_text.strip()
