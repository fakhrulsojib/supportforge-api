"""Text extraction from uploaded documents.

Supports PDF (via pymupdf/fitz), Markdown, CSV, and plain text files.
Handles encoding errors gracefully with UTF-8 → latin-1 fallback.
"""

from __future__ import annotations

import csv
import io

import structlog

logger = structlog.get_logger(__name__)

# Import fitz (pymupdf) at module level so it can be easily mocked in tests
try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None

# Supported file types for extraction
_SUPPORTED_TYPES: set[str] = {"pdf", "md", "csv", "txt"}


class TextExtractionError(Exception):
    """Raised when text extraction from a document fails."""

    def __init__(self, message: str = "Text extraction failed") -> None:
        self.message = message
        super().__init__(self.message)


class TextExtractor:
    """Extracts text content from various file formats.

    Static methods handle each file type. All methods raise
    ``TextExtractionError`` on failure or empty content.
    """

    @staticmethod
    def extract(content: bytes, file_type: str) -> str:
        """Extract text from file content based on file type.

        Args:
            content: Raw file bytes.
            file_type: Lowercase file extension (e.g., "pdf", "md", "csv", "txt").

        Returns:
            Extracted text content.

        Raises:
            TextExtractionError: If extraction fails or yields no text.
        """
        if file_type not in _SUPPORTED_TYPES:
            msg = f"Unsupported file type '{file_type}'. Supported: {', '.join(sorted(_SUPPORTED_TYPES))}"
            raise TextExtractionError(msg)

        extractors = {
            "pdf": TextExtractor._extract_pdf,
            "md": TextExtractor._extract_text,
            "csv": TextExtractor._extract_csv,
            "txt": TextExtractor._extract_text,
        }

        text = extractors[file_type](content)

        if not text or not text.strip():
            raise TextExtractionError("No text content extracted from document")

        return text.strip()

    @staticmethod
    def _extract_pdf(content: bytes) -> str:
        """Extract text from a PDF file using pymupdf (fitz).

        Args:
            content: Raw PDF bytes.

        Returns:
            Concatenated text from all pages.

        Raises:
            TextExtractionError: If the PDF cannot be opened or parsed.
        """
        try:
            with fitz.open(stream=content, filetype="pdf") as doc:
                pages: list[str] = []
                for page in doc:
                    page_text = page.get_text()
                    if page_text and page_text.strip():
                        pages.append(page_text.strip())
                return "\n\n".join(pages)
        except TextExtractionError:
            raise
        except Exception as e:
            msg = f"Failed to extract text from PDF: {e}"
            raise TextExtractionError(msg) from e

    @staticmethod
    def _extract_text(content: bytes) -> str:
        """Extract text from a UTF-8 text file (txt or md).

        Falls back to latin-1 encoding if UTF-8 decoding fails.

        Args:
            content: Raw text file bytes.

        Returns:
            Decoded text string.
        """
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("utf8_decode_failed_falling_back_to_latin1")
            return content.decode("latin-1")

    @staticmethod
    def _extract_csv(content: bytes) -> str:
        """Extract text from a CSV file by concatenating all rows.

        Each row is joined with commas and rows are separated by newlines,
        producing a readable text representation.

        Args:
            content: Raw CSV file bytes.

        Returns:
            Text representation of CSV content.
        """
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("utf8_decode_failed_falling_back_to_latin1")
            text = content.decode("latin-1")

        reader = csv.reader(io.StringIO(text))
        rows: list[str] = []
        for row in reader:
            if row:
                rows.append(", ".join(row))
        return "\n".join(rows)
