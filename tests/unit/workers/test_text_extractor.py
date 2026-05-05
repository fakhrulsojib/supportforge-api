"""Unit tests for text extraction from uploaded documents.

Covers:
    - PDF text extraction (mocked fitz)
    - Markdown text extraction
    - CSV text extraction
    - Plain text extraction
    - Empty file handling
    - Corrupted PDF handling
    - Unsupported file type rejection
    - Encoding error handling
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.workers.text_extractor import TextExtractor, TextExtractionError


# ── PDF Extraction ───────────────────────────────────────────────


class TestPDFExtraction:
    """Tests for PDF file text extraction."""

    @patch("app.workers.text_extractor.fitz")
    def test_extract_pdf_single_page(self, mock_fitz: MagicMock) -> None:
        """PDF with a single page returns extracted text."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Hello from page 1"
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = mock_doc

        result = TextExtractor.extract(b"fake-pdf-bytes", "pdf")

        assert result == "Hello from page 1"
        mock_fitz.open.assert_called_once()

    @patch("app.workers.text_extractor.fitz")
    def test_extract_pdf_multiple_pages(self, mock_fitz: MagicMock) -> None:
        """PDF with multiple pages concatenates text with newlines."""
        pages = []
        for i in range(3):
            page = MagicMock()
            page.get_text.return_value = f"Page {i + 1} content"
            pages.append(page)

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter(pages))
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = mock_doc

        result = TextExtractor.extract(b"fake-pdf-bytes", "pdf")

        assert "Page 1 content" in result
        assert "Page 2 content" in result
        assert "Page 3 content" in result

    @patch("app.workers.text_extractor.fitz")
    def test_extract_pdf_empty_pages(self, mock_fitz: MagicMock) -> None:
        """PDF where all pages return empty text raises error."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "   "
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = mock_doc

        with pytest.raises(TextExtractionError, match="No text content"):
            TextExtractor.extract(b"fake-pdf-bytes", "pdf")

    @patch("app.workers.text_extractor.fitz")
    def test_extract_pdf_corrupted(self, mock_fitz: MagicMock) -> None:
        """Corrupted PDF raises TextExtractionError."""
        mock_fitz.open.side_effect = Exception("cannot open broken document")

        with pytest.raises(TextExtractionError, match="Failed to extract text from PDF"):
            TextExtractor.extract(b"corrupted-bytes", "pdf")


# ── Markdown Extraction ──────────────────────────────────────────


class TestMarkdownExtraction:
    """Tests for Markdown file text extraction."""

    def test_extract_markdown_simple(self) -> None:
        """Markdown content extracted as UTF-8 text."""
        content = b"# Heading\n\nSome paragraph text.\n\n- Item 1\n- Item 2"
        result = TextExtractor.extract(content, "md")
        assert "# Heading" in result
        assert "Some paragraph text." in result

    def test_extract_markdown_unicode(self) -> None:
        """Markdown with unicode characters extracted correctly."""
        content = "# Héading with ünïcödë\n\nParagraph with émojis 🎉".encode("utf-8")
        result = TextExtractor.extract(content, "md")
        assert "ünïcödë" in result
        assert "🎉" in result

    def test_extract_markdown_empty(self) -> None:
        """Empty markdown raises TextExtractionError."""
        with pytest.raises(TextExtractionError, match="No text content"):
            TextExtractor.extract(b"   \n  \n  ", "md")


# ── CSV Extraction ───────────────────────────────────────────────


class TestCSVExtraction:
    """Tests for CSV file text extraction."""

    def test_extract_csv_simple(self) -> None:
        """CSV rows extracted as readable text."""
        content = b"name,age,city\nAlice,30,NYC\nBob,25,LA"
        result = TextExtractor.extract(content, "csv")
        assert "Alice" in result
        assert "Bob" in result
        assert "name" in result

    def test_extract_csv_with_commas_in_fields(self) -> None:
        """CSV with quoted fields containing commas."""
        content = b'question,answer\n"What is 1+1?","It is 2, obviously"\n"Why?","Because math"'
        result = TextExtractor.extract(content, "csv")
        assert "It is 2, obviously" in result
        assert "Because math" in result

    def test_extract_csv_empty(self) -> None:
        """Empty CSV raises TextExtractionError."""
        with pytest.raises(TextExtractionError, match="No text content"):
            TextExtractor.extract(b"", "csv")

    def test_extract_csv_headers_only(self) -> None:
        """CSV with only headers and no data rows still returns header text."""
        content = b"name,age,city\n"
        result = TextExtractor.extract(content, "csv")
        assert "name" in result


# ── Plain Text Extraction ────────────────────────────────────────


class TestTextExtraction:
    """Tests for plain text file extraction."""

    def test_extract_txt_simple(self) -> None:
        """Plain text extracted directly."""
        content = b"Hello, this is a plain text document."
        result = TextExtractor.extract(content, "txt")
        assert result == "Hello, this is a plain text document."

    def test_extract_txt_unicode(self) -> None:
        """Plain text with UTF-8 unicode."""
        content = "Héllo wörld".encode("utf-8")
        result = TextExtractor.extract(content, "txt")
        assert result == "Héllo wörld"

    def test_extract_txt_empty(self) -> None:
        """Empty text file raises TextExtractionError."""
        with pytest.raises(TextExtractionError, match="No text content"):
            TextExtractor.extract(b"", "txt")

    def test_extract_txt_whitespace_only(self) -> None:
        """Whitespace-only text file raises TextExtractionError."""
        with pytest.raises(TextExtractionError, match="No text content"):
            TextExtractor.extract(b"   \n\t\n   ", "txt")


# ── Unsupported Types ────────────────────────────────────────────


class TestUnsupportedTypes:
    """Tests for unsupported file type rejection."""

    def test_extract_unsupported_type_docx(self) -> None:
        """Unsupported file type raises TextExtractionError."""
        with pytest.raises(TextExtractionError, match="Unsupported file type"):
            TextExtractor.extract(b"some bytes", "docx")

    def test_extract_unsupported_type_xlsx(self) -> None:
        """Another unsupported type raises TextExtractionError."""
        with pytest.raises(TextExtractionError, match="Unsupported file type"):
            TextExtractor.extract(b"some bytes", "xlsx")

    def test_extract_empty_type(self) -> None:
        """Empty file type raises TextExtractionError."""
        with pytest.raises(TextExtractionError, match="Unsupported file type"):
            TextExtractor.extract(b"some bytes", "")


# ── Encoding Errors ──────────────────────────────────────────────


class TestEncodingErrors:
    """Tests for encoding error handling."""

    def test_extract_txt_invalid_encoding(self) -> None:
        """Invalid UTF-8 bytes fall back to latin-1 decoding."""
        # 0xff 0xfe is not valid UTF-8
        content = b"\xff\xfeSome text"
        result = TextExtractor.extract(content, "txt")
        # Should not raise — falls back to latin-1
        assert "Some text" in result

    def test_extract_md_invalid_encoding(self) -> None:
        """Invalid UTF-8 in markdown falls back to latin-1."""
        content = b"\xff\xfe# Heading"
        result = TextExtractor.extract(content, "md")
        assert "Heading" in result
