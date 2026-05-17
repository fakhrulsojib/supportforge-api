"""Unit tests for the IngestionService domain service.

Covers:
    - Successful end-to-end ingestion (extract → chunk → contextualise → embed → store → persist)
    - Status transitions (PENDING → PROCESSING → READY)
    - Partial embedding failure → rollback, status FAILED
    - Vector store failure → rollback, status FAILED
    - Empty text extraction → status FAILED
    - Chunk count persisted correctly
    - ChromaDB IDs match document_chunks.chroma_id
    - Contextual retrieval integration (LLM provider present)
    - Graceful fallback when LLM provider is absent
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.exceptions import IngestionError
from app.domain.models.document import Document, DocumentChunk
from app.domain.models.enums import DocumentStatus
from app.domain.services.ingestion_service import IngestionService


@pytest.fixture
def mock_document_repo() -> AsyncMock:
    """Create a mock DocumentRepository."""
    repo = AsyncMock()
    repo.update_status = AsyncMock()
    repo.create_chunk = AsyncMock(side_effect=lambda chunk: chunk)
    repo.delete_chunks_by_document = AsyncMock(return_value=0)
    return repo


@pytest.fixture
def mock_embedding_service() -> AsyncMock:
    """Create a mock EmbeddingService."""
    service = AsyncMock()
    service.embed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    return service


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    """Create a mock VectorStore."""
    store = AsyncMock()
    store.add_documents = AsyncMock()
    return store


@pytest.fixture
def mock_llm_provider() -> AsyncMock:
    """Create a mock LLMProvider for contextual retrieval."""
    provider = AsyncMock()
    provider.generate = AsyncMock(
        return_value="This chunk is from the test document about general support topics."
    )
    return provider


@pytest.fixture
def sample_document() -> Document:
    """Create a sample document for testing."""
    return Document(
        id="doc-123",
        tenant_id="tenant-abc",
        filename="test.txt",
        file_type="txt",
        chunk_count=0,
        status=DocumentStatus.PENDING,
        uploaded_by="user-xyz",
    )


@pytest.fixture
def ingestion_service(
    mock_document_repo: AsyncMock,
    mock_embedding_service: AsyncMock,
    mock_vector_store: AsyncMock,
) -> IngestionService:
    """Create an IngestionService with mocked dependencies (no LLM provider)."""
    return IngestionService(
        document_repo=mock_document_repo,
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
    )


@pytest.fixture
def ingestion_service_with_llm(
    mock_document_repo: AsyncMock,
    mock_embedding_service: AsyncMock,
    mock_vector_store: AsyncMock,
    mock_llm_provider: AsyncMock,
) -> IngestionService:
    """Create an IngestionService with LLM provider for contextual retrieval."""
    return IngestionService(
        document_repo=mock_document_repo,
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        llm_provider=mock_llm_provider,
    )


# ── Successful Ingestion ─────────────────────────────────────────


class TestSuccessfulIngestion:
    """Tests for the happy path of document ingestion."""

    @patch("app.domain.services.ingestion_service.TextExtractor")
    async def test_successful_ingestion_updates_status_to_ready(
        self,
        mock_extractor: MagicMock,
        ingestion_service: IngestionService,
        mock_document_repo: AsyncMock,
        sample_document: Document,
    ) -> None:
        """Successful ingestion transitions status PENDING → PROCESSING → READY."""
        mock_extractor.extract.return_value = "Some document text content for testing."
        mock_document_repo.update_status.return_value = sample_document

        await ingestion_service.process_document(
            document=sample_document,
            file_content=b"Some document text content for testing.",
        )

        # Should update to PROCESSING first, then READY
        calls = mock_document_repo.update_status.call_args_list
        assert len(calls) >= 2
        assert calls[0].kwargs["status"] == DocumentStatus.PROCESSING or calls[0].args[1] == DocumentStatus.PROCESSING
        # Last call should be READY with chunk count
        last_call = calls[-1]
        assert DocumentStatus.READY in (last_call.args[1] if len(last_call.args) > 1 else last_call.kwargs["status"],)

    @patch("app.domain.services.ingestion_service.TextExtractor")
    async def test_successful_ingestion_creates_chunks_in_db(
        self,
        mock_extractor: MagicMock,
        ingestion_service: IngestionService,
        mock_document_repo: AsyncMock,
        mock_embedding_service: AsyncMock,
        sample_document: Document,
    ) -> None:
        """Successful ingestion creates chunk records in the database."""
        mock_extractor.extract.return_value = "First chunk of text. " * 50 + "\n\n" + "Second chunk of text. " * 50
        mock_document_repo.update_status.return_value = sample_document
        mock_embedding_service.embed_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]

        await ingestion_service.process_document(
            document=sample_document,
            file_content=b"text",
        )

        # Should have created chunks in DB
        assert mock_document_repo.create_chunk.call_count > 0

    @patch("app.domain.services.ingestion_service.TextExtractor")
    async def test_successful_ingestion_stores_in_vector_db(
        self,
        mock_extractor: MagicMock,
        ingestion_service: IngestionService,
        mock_document_repo: AsyncMock,
        mock_vector_store: AsyncMock,
        mock_embedding_service: AsyncMock,
        sample_document: Document,
    ) -> None:
        """Successful ingestion stores embeddings in the vector database."""
        mock_extractor.extract.return_value = "Document content for embedding."
        mock_document_repo.update_status.return_value = sample_document
        mock_embedding_service.embed_batch.return_value = [[0.1, 0.2]]

        await ingestion_service.process_document(
            document=sample_document,
            file_content=b"text",
        )

        mock_vector_store.add_documents.assert_called_once()
        call_args = mock_vector_store.add_documents.call_args
        assert call_args.kwargs["tenant_id"] == "tenant-abc"

    @patch("app.domain.services.ingestion_service.TextExtractor")
    async def test_chunk_count_matches_created_chunks(
        self,
        mock_extractor: MagicMock,
        ingestion_service: IngestionService,
        mock_document_repo: AsyncMock,
        mock_embedding_service: AsyncMock,
        sample_document: Document,
    ) -> None:
        """The chunk_count in status update matches the actual number of chunks created."""
        mock_extractor.extract.return_value = "Some text content."
        mock_document_repo.update_status.return_value = sample_document
        mock_embedding_service.embed_batch.return_value = [[0.1, 0.2]]

        await ingestion_service.process_document(
            document=sample_document,
            file_content=b"text",
        )

        # The final update_status call should include the correct chunk count
        last_update = mock_document_repo.update_status.call_args_list[-1]
        chunk_count = last_update.kwargs.get("chunk_count", last_update.args[2] if len(last_update.args) > 2 else 0)
        assert chunk_count == mock_document_repo.create_chunk.call_count

    @patch("app.domain.services.ingestion_service.TextExtractor")
    async def test_chroma_ids_match_chunk_records(
        self,
        mock_extractor: MagicMock,
        ingestion_service: IngestionService,
        mock_document_repo: AsyncMock,
        mock_embedding_service: AsyncMock,
        mock_vector_store: AsyncMock,
        sample_document: Document,
    ) -> None:
        """ChromaDB IDs used for vector storage match the chroma_id on DB chunks."""
        mock_extractor.extract.return_value = "Content for chunking."
        mock_document_repo.update_status.return_value = sample_document
        mock_embedding_service.embed_batch.return_value = [[0.1]]

        await ingestion_service.process_document(
            document=sample_document,
            file_content=b"text",
        )

        # Get IDs sent to vector store
        vector_call = mock_vector_store.add_documents.call_args
        vector_ids = vector_call.kwargs["ids"]

        # Get IDs from created chunks
        chunk_calls = mock_document_repo.create_chunk.call_args_list
        chunk_chroma_ids = [call.args[0].chroma_id for call in chunk_calls]

        assert set(vector_ids) == set(chunk_chroma_ids)


# ── Contextual Retrieval ─────────────────────────────────────────


class TestContextualRetrieval:
    """Tests for Anthropic's contextual retrieval technique."""

    @patch("app.domain.services.ingestion_service.TextExtractor")
    async def test_chunks_are_contextualised_when_llm_provider_present(
        self,
        mock_extractor: MagicMock,
        ingestion_service_with_llm: IngestionService,
        mock_document_repo: AsyncMock,
        mock_embedding_service: AsyncMock,
        mock_vector_store: AsyncMock,
        mock_llm_provider: AsyncMock,
        sample_document: Document,
    ) -> None:
        """When LLM provider is present, chunks should be contextualised before embedding."""
        mock_extractor.extract.return_value = "Document content for contextualisation."
        mock_document_repo.update_status.return_value = sample_document
        mock_embedding_service.embed_batch.return_value = [[0.1, 0.2]]

        await ingestion_service_with_llm.process_document(
            document=sample_document,
            file_content=b"text",
        )

        # LLM should have been called for context generation
        assert mock_llm_provider.generate.call_count > 0

        # The text sent to embed_batch should contain the contextualised prefix
        embed_call = mock_embedding_service.embed_batch.call_args
        embedded_texts = embed_call.args[0]
        for text in embedded_texts:
            assert "This chunk is from the test document" in text

    @patch("app.domain.services.ingestion_service.TextExtractor")
    async def test_chunks_stored_with_contextualised_text(
        self,
        mock_extractor: MagicMock,
        ingestion_service_with_llm: IngestionService,
        mock_document_repo: AsyncMock,
        mock_embedding_service: AsyncMock,
        mock_llm_provider: AsyncMock,
        sample_document: Document,
    ) -> None:
        """DB chunk records should contain the contextualised text."""
        mock_extractor.extract.return_value = "Short content."
        mock_document_repo.update_status.return_value = sample_document
        mock_embedding_service.embed_batch.return_value = [[0.1]]

        await ingestion_service_with_llm.process_document(
            document=sample_document,
            file_content=b"text",
        )

        # Check the chunk content stored in DB
        chunk_calls = mock_document_repo.create_chunk.call_args_list
        for call in chunk_calls:
            chunk: DocumentChunk = call.args[0]
            assert "This chunk is from the test document" in chunk.content

    @patch("app.domain.services.ingestion_service.TextExtractor")
    async def test_ingestion_works_without_llm_provider(
        self,
        mock_extractor: MagicMock,
        ingestion_service: IngestionService,
        mock_document_repo: AsyncMock,
        mock_embedding_service: AsyncMock,
        sample_document: Document,
    ) -> None:
        """Ingestion should work fine without LLM provider (no contextualisation)."""
        mock_extractor.extract.return_value = "Raw content without context."
        mock_document_repo.update_status.return_value = sample_document
        mock_embedding_service.embed_batch.return_value = [[0.1]]

        await ingestion_service.process_document(
            document=sample_document,
            file_content=b"text",
        )

        # Should complete successfully
        last_update = mock_document_repo.update_status.call_args_list[-1]
        assert DocumentStatus.READY in (
            last_update.args[1] if len(last_update.args) > 1 else last_update.kwargs["status"],
        )

    @patch("app.domain.services.ingestion_service.TextExtractor")
    async def test_contextualisation_failure_gracefully_degrades(
        self,
        mock_extractor: MagicMock,
        ingestion_service_with_llm: IngestionService,
        mock_document_repo: AsyncMock,
        mock_embedding_service: AsyncMock,
        mock_llm_provider: AsyncMock,
        sample_document: Document,
    ) -> None:
        """If LLM context generation fails, original chunks should be used."""
        mock_extractor.extract.return_value = "Content for fallback test."
        mock_document_repo.update_status.return_value = sample_document
        mock_embedding_service.embed_batch.return_value = [[0.1]]
        # Simulate LLM failure — contextualizer catches and returns raw chunk
        mock_llm_provider.generate.side_effect = Exception("LLM unavailable")

        await ingestion_service_with_llm.process_document(
            document=sample_document,
            file_content=b"text",
        )

        # Should still complete successfully with raw chunks
        last_update = mock_document_repo.update_status.call_args_list[-1]
        assert DocumentStatus.READY in (
            last_update.args[1] if len(last_update.args) > 1 else last_update.kwargs["status"],
        )


# ── Failure Handling ─────────────────────────────────────────────


class TestFailureHandling:
    """Tests for error handling and rollback during ingestion."""

    @patch("app.domain.services.ingestion_service.TextExtractor")
    async def test_text_extraction_failure_sets_status_failed(
        self,
        mock_extractor: MagicMock,
        ingestion_service: IngestionService,
        mock_document_repo: AsyncMock,
        sample_document: Document,
    ) -> None:
        """Text extraction failure sets document status to FAILED."""
        from app.rag.text_extractor import TextExtractionError

        mock_extractor.extract.side_effect = TextExtractionError("Cannot extract text")
        mock_document_repo.update_status.return_value = sample_document

        with pytest.raises(IngestionError, match="Cannot extract text"):
            await ingestion_service.process_document(
                document=sample_document,
                file_content=b"bad content",
            )

        # Status should be set to FAILED
        last_update = mock_document_repo.update_status.call_args_list[-1]
        assert DocumentStatus.FAILED in (
            last_update.args[1] if len(last_update.args) > 1 else last_update.kwargs["status"],
        )

    @patch("app.domain.services.ingestion_service.TextExtractor")
    async def test_embedding_failure_rolls_back_and_sets_failed(
        self,
        mock_extractor: MagicMock,
        ingestion_service: IngestionService,
        mock_document_repo: AsyncMock,
        mock_embedding_service: AsyncMock,
        sample_document: Document,
    ) -> None:
        """Embedding failure triggers rollback of partial chunks and sets FAILED."""
        mock_extractor.extract.return_value = "Content for embedding."
        mock_embedding_service.embed_batch.side_effect = Exception("Ollama connection refused")
        mock_document_repo.update_status.return_value = sample_document

        with pytest.raises(IngestionError, match="connection refused"):
            await ingestion_service.process_document(
                document=sample_document,
                file_content=b"text",
            )

        # Should clean up any partial chunks
        mock_document_repo.delete_chunks_by_document.assert_called_with(sample_document.id)
        # Status set to FAILED
        last_update = mock_document_repo.update_status.call_args_list[-1]
        assert DocumentStatus.FAILED in (
            last_update.args[1] if len(last_update.args) > 1 else last_update.kwargs["status"],
        )

    @patch("app.domain.services.ingestion_service.TextExtractor")
    async def test_vector_store_failure_rolls_back_and_sets_failed(
        self,
        mock_extractor: MagicMock,
        ingestion_service: IngestionService,
        mock_document_repo: AsyncMock,
        mock_embedding_service: AsyncMock,
        mock_vector_store: AsyncMock,
        sample_document: Document,
    ) -> None:
        """Vector store failure triggers rollback and sets FAILED."""
        mock_extractor.extract.return_value = "Content for vector store."
        mock_embedding_service.embed_batch.return_value = [[0.1]]
        mock_vector_store.add_documents.side_effect = Exception("ChromaDB unavailable")
        mock_document_repo.update_status.return_value = sample_document

        with pytest.raises(IngestionError, match="ChromaDB unavailable"):
            await ingestion_service.process_document(
                document=sample_document,
                file_content=b"text",
            )

        mock_document_repo.delete_chunks_by_document.assert_called_with(sample_document.id)
        last_update = mock_document_repo.update_status.call_args_list[-1]
        assert DocumentStatus.FAILED in (
            last_update.args[1] if len(last_update.args) > 1 else last_update.kwargs["status"],
        )

    @patch("app.domain.services.ingestion_service.TextExtractor")
    async def test_empty_extraction_result_sets_failed(
        self,
        mock_extractor: MagicMock,
        ingestion_service: IngestionService,
        mock_document_repo: AsyncMock,
        sample_document: Document,
    ) -> None:
        """Empty text extraction result sets FAILED status."""
        from app.rag.text_extractor import TextExtractionError

        mock_extractor.extract.side_effect = TextExtractionError("No text content extracted")
        mock_document_repo.update_status.return_value = sample_document

        with pytest.raises(IngestionError):
            await ingestion_service.process_document(
                document=sample_document,
                file_content=b"",
            )

        last_update = mock_document_repo.update_status.call_args_list[-1]
        assert DocumentStatus.FAILED in (
            last_update.args[1] if len(last_update.args) > 1 else last_update.kwargs["status"],
        )


# ── Edge Cases ───────────────────────────────────────────────────


class TestEdgeCases:
    """Tests for boundary conditions and edge cases."""

    @patch("app.domain.services.ingestion_service.TextExtractor")
    async def test_very_short_text_produces_single_chunk(
        self,
        mock_extractor: MagicMock,
        ingestion_service: IngestionService,
        mock_document_repo: AsyncMock,
        mock_embedding_service: AsyncMock,
        sample_document: Document,
    ) -> None:
        """Very short text produces a single chunk."""
        mock_extractor.extract.return_value = "Short text."
        mock_document_repo.update_status.return_value = sample_document
        mock_embedding_service.embed_batch.return_value = [[0.1]]

        await ingestion_service.process_document(
            document=sample_document,
            file_content=b"Short text.",
        )

        assert mock_document_repo.create_chunk.call_count == 1

    @patch("app.domain.services.ingestion_service.TextExtractor")
    async def test_chunk_metadata_includes_document_info(
        self,
        mock_extractor: MagicMock,
        ingestion_service: IngestionService,
        mock_document_repo: AsyncMock,
        mock_embedding_service: AsyncMock,
        mock_vector_store: AsyncMock,
        sample_document: Document,
    ) -> None:
        """Metadata sent to vector store includes document and tenant info."""
        mock_extractor.extract.return_value = "Content with metadata."
        mock_document_repo.update_status.return_value = sample_document
        mock_embedding_service.embed_batch.return_value = [[0.1]]

        await ingestion_service.process_document(
            document=sample_document,
            file_content=b"text",
        )

        vector_call = mock_vector_store.add_documents.call_args
        metadatas = vector_call.kwargs["metadatas"]
        assert len(metadatas) > 0
        assert metadatas[0]["document_id"] == "doc-123"
        assert metadatas[0]["filename"] == "test.txt"


# ── Section Merging ──────────────────────────────────────────────


class TestSectionMerging:
    """Tests for _merge_small_sections — preventing over-chunking."""

    @pytest.fixture
    def service(self) -> IngestionService:
        """Service with 500-token chunk size."""
        return IngestionService(
            document_repo=AsyncMock(),
            embedding_service=AsyncMock(),
            vector_store=AsyncMock(),
            chunk_size=500,
            chunk_overlap=75,
        )

    @pytest.fixture
    def small_service(self) -> IngestionService:
        """Service with very small chunk size for easier testing."""
        return IngestionService(
            document_repo=AsyncMock(),
            embedding_service=AsyncMock(),
            vector_store=AsyncMock(),
            chunk_size=50,
            chunk_overlap=10,
        )

    def _make_section(
        self, content: str, heading_path: str = "", heading_level: int = 0
    ) -> MagicMock:
        """Helper to create a mock MarkdownSection."""
        section = MagicMock()
        section.content = content
        section.heading_path = heading_path
        section.heading_level = heading_level
        return section

    def test_empty_sections_returns_empty(self, service: IngestionService) -> None:
        """No sections → no groups."""
        assert service._merge_small_sections([]) == []

    def test_single_small_section_produces_one_group(
        self, service: IngestionService
    ) -> None:
        """A single small section produces exactly one group."""
        sections = [self._make_section("Short text.", "Title")]
        groups = service._merge_small_sections(sections)
        assert len(groups) == 1
        assert "Short text." in groups[0][0]
        assert groups[0][1] == "Title"

    def test_small_sections_merged_together(
        self, service: IngestionService
    ) -> None:
        """Multiple small sections below chunk_size are merged into one group."""
        sections = [
            self._make_section("Section one.", "A"),
            self._make_section("Section two.", "A > B"),
            self._make_section("Section three.", "A > C"),
        ]
        groups = service._merge_small_sections(sections)
        assert len(groups) == 1
        assert "Section one." in groups[0][0]
        assert "Section two." in groups[0][0]
        assert "Section three." in groups[0][0]

    def test_heading_path_from_first_section_in_group(
        self, service: IngestionService
    ) -> None:
        """Merged group uses heading_path from the first section."""
        sections = [
            self._make_section("First.", "Root > First"),
            self._make_section("Second.", "Root > Second"),
        ]
        groups = service._merge_small_sections(sections)
        assert len(groups) == 1
        assert groups[0][1] == "Root > First"

    def test_large_section_stays_alone(self, small_service: IngestionService) -> None:
        """A section exceeding chunk_size stays in its own group."""
        # 50-token limit; create one big and one small section
        big_text = " ".join(["word"] * 100)  # ~100 tokens
        sections = [
            self._make_section("Small.", "A"),
            self._make_section(big_text, "B"),
            self._make_section("Also small.", "C"),
        ]
        groups = small_service._merge_small_sections(sections)
        # Big section should force a split
        assert len(groups) >= 2
        # The big text should be in its own group
        big_groups = [g for g in groups if "word" in g[0] and len(g[0]) > 200]
        assert len(big_groups) == 1

    def test_overflow_creates_new_group(
        self, small_service: IngestionService
    ) -> None:
        """When accumulated tokens exceed chunk_size, a new group starts."""
        sections = [
            self._make_section(" ".join(["alpha"] * 30), "G1"),
            self._make_section(" ".join(["beta"] * 30), "G2"),
            self._make_section(" ".join(["gamma"] * 30), "G3"),
        ]
        groups = small_service._merge_small_sections(sections)
        # With 50-token limit and ~30 tokens each, should get 2-3 groups
        assert len(groups) >= 2
        # Each group heading should be from its first section
        assert groups[0][1] == "G1"

    def test_sections_joined_with_double_newline(
        self, service: IngestionService
    ) -> None:
        """Merged sections are joined with \\n\\n separator."""
        sections = [
            self._make_section("Part A.", "X"),
            self._make_section("Part B.", "X > Y"),
        ]
        groups = service._merge_small_sections(sections)
        assert groups[0][0] == "Part A.\n\nPart B."

    def test_empty_heading_path_preserved(
        self, service: IngestionService
    ) -> None:
        """Sections with no heading path (preamble) are handled correctly."""
        sections = [
            self._make_section("Preamble text.", ""),
            self._make_section("After heading.", "Title"),
        ]
        groups = service._merge_small_sections(sections)
        assert len(groups) == 1
        assert groups[0][1] == ""  # First section had no heading

    def test_many_tiny_sections_merge_efficiently(
        self, service: IngestionService
    ) -> None:
        """31 tiny sections (like the warranty doc) merge into far fewer groups."""
        sections = [
            self._make_section(f"Section {i} with some text.", f"Path {i}")
            for i in range(31)
        ]
        groups = service._merge_small_sections(sections)
        # 31 sections of ~8 tokens each = ~248 tokens total
        # Should fit in 1 group with 500-token limit
        assert len(groups) < 31
        assert len(groups) <= 3  # Conservative: should be 1-2

    def test_single_section_exceeding_chunk_size(
        self, small_service: IngestionService
    ) -> None:
        """A single section larger than chunk_size still produces one group.

        The RecursiveChunker (not the merger) handles splitting it further.
        """
        big_text = " ".join(["word"] * 200)
        sections = [self._make_section(big_text, "BigSection")]
        groups = small_service._merge_small_sections(sections)
        assert len(groups) == 1
        assert groups[0][1] == "BigSection"


# ── Markdown Chunking Integration ────────────────────────────────


class TestMarkdownChunking:
    """Tests for _chunk_markdown with section merging."""

    @pytest.fixture
    def service(self) -> IngestionService:
        """Service with 500-token chunk size."""
        return IngestionService(
            document_repo=AsyncMock(),
            embedding_service=AsyncMock(),
            vector_store=AsyncMock(),
            chunk_size=500,
            chunk_overlap=75,
        )

    def test_markdown_with_many_small_headings_merged(
        self, service: IngestionService
    ) -> None:
        """Document with many small heading sections produces reasonable chunks."""
        md = "\n\n".join(
            f"## Section {i}\n\nShort content for section {i}."
            for i in range(20)
        )
        chunks = service._chunk_markdown(md, "doc-1", "test.md")
        # 20 sections of ~12 tokens each = ~240 tokens
        # Should produce far fewer chunks than 20
        assert len(chunks) < 20
        assert len(chunks) >= 1

    def test_markdown_no_headings_falls_back(
        self, service: IngestionService
    ) -> None:
        """Document with no headings uses plain RecursiveChunker."""
        text = "Just plain text without any headings. " * 10
        chunks = service._chunk_markdown(text, "doc-1", "plain.md")
        assert len(chunks) >= 1
        # No heading_path metadata
        assert "heading_path" not in chunks[0].metadata

    def test_chunk_indices_globally_sequential(
        self, service: IngestionService
    ) -> None:
        """Chunk indices are sequential across merged groups."""
        md = (
            "# Title\n\nIntro.\n\n"
            + "\n\n".join(
                f"## S{i}\n\nContent {i}." for i in range(10)
            )
        )
        chunks = service._chunk_markdown(md, "doc-1", "test.md")
        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_metadata_has_document_fields(
        self, service: IngestionService
    ) -> None:
        """Every chunk has document_id, filename, file_type metadata."""
        md = "# Title\n\nSome content."
        chunks = service._chunk_markdown(md, "doc-42", "policy.md")
        for chunk in chunks:
            assert chunk.metadata["document_id"] == "doc-42"
            assert chunk.metadata["filename"] == "policy.md"
            assert chunk.metadata["file_type"] == "md"

    def test_heading_path_preserved_in_metadata(
        self, service: IngestionService
    ) -> None:
        """Merged groups carry heading_path from their first section."""
        md = "# Shipping\n\n## Methods\n\nStandard, Express.\n\n## Costs\n\n$5, $12."
        chunks = service._chunk_markdown(md, "doc-1", "ship.md")
        # All content fits in one chunk, heading from first section
        paths = [c.metadata.get("heading_path", "") for c in chunks]
        assert any("Shipping" in p for p in paths)

    def test_large_section_still_split_by_chunker(
        self, service: IngestionService
    ) -> None:
        """A section exceeding chunk_size is still split by RecursiveChunker."""
        # Create a single section with >500 tokens
        big_content = " ".join(["This is a test sentence."] * 200)
        md = f"# Big Section\n\n{big_content}"
        chunks = service._chunk_markdown(md, "doc-1", "big.md")
        assert len(chunks) > 1  # Must be split by the chunker

    def test_empty_markdown_returns_empty(
        self, service: IngestionService
    ) -> None:
        """Empty/whitespace markdown produces no chunks."""
        chunks = service._chunk_markdown("", "doc-1", "empty.md")
        assert chunks == []

    def test_realistic_faq_document(self, service: IngestionService) -> None:
        """Realistic FAQ document with many short Q&A sections."""
        sections = [
            "# FAQ\n\nFrequently asked questions about our service.",
        ]
        for i in range(15):
            sections.append(
                f"## Question {i+1}\n\n"
                f"**Q:** How do I do thing number {i+1}?\n\n"
                f"**A:** You can do thing {i+1} by following these steps: "
                f"first, open the app. Then navigate to settings."
            )
        md = "\n\n".join(sections)
        chunks = service._chunk_markdown(md, "doc-faq", "faq.md")
        # 15 Q&A sections of ~40 tokens each = ~600 tokens
        # Should produce ~2 chunks, definitely not 15+
        assert len(chunks) < 10
        assert len(chunks) >= 1

    def test_merged_group_tokens_do_not_exceed_chunk_size(
        self, service: IngestionService
    ) -> None:
        """Verify merged group text never exceeds chunk_size in tokens.

        The merge logic must account for separator tokens so the
        RecursiveChunker doesn't need to split every group.
        """
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        # Create sections that are just under half of chunk_size each
        # Two should fit, three should not
        word_block = " ".join(["evaluate"] * 55)  # ~55 tokens
        md = "\n\n".join(
            f"## Section {i}\n\n{word_block}" for i in range(8)
        )
        chunks = service._chunk_markdown(md, "doc-1", "test.md")
        for chunk in chunks:
            tokens = len(enc.encode(chunk.content))
            # Allow up to chunk_size + overlap (chunker may add overlap)
            assert tokens <= 500 + 75, (
                f"Chunk {chunk.index} has {tokens} tokens, exceeds 575"
            )

    def test_preamble_text_merged_with_first_heading(
        self, service: IngestionService
    ) -> None:
        """Text before the first heading merges with subsequent small sections."""
        md = (
            "This document covers our policies.\n\n"
            "# Policies\n\nGeneral info.\n\n"
            "## Details\n\nMore details here."
        )
        chunks = service._chunk_markdown(md, "doc-1", "test.md")
        # Everything is small, should all merge into 1 chunk
        assert len(chunks) == 1
        assert "This document covers" in chunks[0].content
        assert "More details here" in chunks[0].content

    def test_all_sections_oversized_no_merge(self) -> None:
        """When every section exceeds chunk_size, no merging happens."""
        svc = IngestionService(
            document_repo=AsyncMock(),
            embedding_service=AsyncMock(),
            vector_store=AsyncMock(),
            chunk_size=50,
            chunk_overlap=10,
        )
        big = " ".join(["word"] * 100)
        md = f"# A\n\n{big}\n\n## B\n\n{big}\n\n## C\n\n{big}"
        chunks = svc._chunk_markdown(md, "doc-1", "test.md")
        # Each section is >50 tokens, so each must become its own group
        # and the RecursiveChunker will further split them
        assert len(chunks) > 3

    def test_alternating_large_and_small_sections(self) -> None:
        """Large sections flush; small sections after them start fresh groups."""
        svc = IngestionService(
            document_repo=AsyncMock(),
            embedding_service=AsyncMock(),
            vector_store=AsyncMock(),
            chunk_size=50,
            chunk_overlap=10,
        )
        big = " ".join(["word"] * 80)
        md = (
            f"## Big1\n\n{big}\n\n"
            "## Small1\n\nTiny.\n\n"
            f"## Big2\n\n{big}\n\n"
            "## Small2\n\nAlso tiny."
        )
        chunks = svc._chunk_markdown(md, "doc-1", "test.md")
        # Big sections can't merge with anything
        # Small sections should merge with adjacent smalls if possible
        assert len(chunks) >= 3

    def test_heading_only_section_contributes_minimal_tokens(
        self, service: IngestionService
    ) -> None:
        """Heading-only sections (like '## Troubleshooting Guides') are tiny."""
        md = (
            "## Overview\n\n"
            "## Section A\n\nContent A here.\n\n"
            "## Section B\n\nContent B here."
        )
        chunks = service._chunk_markdown(md, "doc-1", "test.md")
        # All fits in 1 chunk — heading-only sections add minimal tokens
        assert len(chunks) == 1
        assert "## Overview" in chunks[0].content
        assert "Content A" in chunks[0].content
        assert "Content B" in chunks[0].content

    def test_code_fence_preserved_through_merge(
        self, service: IngestionService
    ) -> None:
        """Code fences inside sections survive the merge process intact."""
        md = (
            "## Config\n\n"
            "```python\nx = 42\n```\n\n"
            "## Usage\n\nCall `run()`."
        )
        chunks = service._chunk_markdown(md, "doc-1", "test.md")
        combined = " ".join(c.content for c in chunks)
        assert "```python" in combined
        assert "x = 42" in combined
        assert "```" in combined

    def test_table_preserved_through_merge(
        self, service: IngestionService
    ) -> None:
        """Tables inside sections survive the merge process intact."""
        md = (
            "## Pricing\n\n"
            "| Plan | Cost |\n|------|------|\n| Free | $0 |\n| Pro | $10 |\n\n"
            "## Notes\n\nSee above."
        )
        chunks = service._chunk_markdown(md, "doc-1", "test.md")
        combined = " ".join(c.content for c in chunks)
        assert "| Free | $0 |" in combined
        assert "| Pro | $10 |" in combined

    def test_chunk_index_metadata_matches_index_field(
        self, service: IngestionService
    ) -> None:
        """chunk.index and chunk.metadata['chunk_index'] are always equal."""
        md = "\n\n".join(
            f"## S{i}\n\n{'Content. ' * 50}" for i in range(5)
        )
        chunks = service._chunk_markdown(md, "doc-1", "test.md")
        for chunk in chunks:
            assert chunk.index == chunk.metadata["chunk_index"]

    def test_realistic_warranty_style_document(
        self, service: IngestionService
    ) -> None:
        """Realistic document with 20+ tiny heading sections (warranty-style).

        Reproduces the over-chunking bug: a structured document with many
        short headings should merge into a handful of well-sized chunks,
        not produce one chunk per heading.
        """
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        md = (
            "# Product Warranty & Support\n\n"
            "**Last Updated:** January 2025\n\n"
            "## Standard Warranty\n\n"
            "All products include a 1-year manufacturer warranty.\n\n"
            "### What's Covered\n\n"
            "- Manufacturing defects\n- Hardware failures\n- Component issues\n\n"
            "### What's NOT Covered\n\n"
            "- Physical damage from drops\n- Water damage\n"
            "- Unauthorized modifications\n- Normal wear and tear\n\n"
            "## Extended Warranty Plans\n\n"
            "| Plan | Duration | Price |\n|------|----------|-------|\n"
            "| Basic | 2 years | $49 |\n| Premium | 3 years | $99 |\n\n"
            "### Basic Plan Details\n\nCovers parts and labor.\n\n"
            "### Premium Plan Details\n\nCovers accidental damage.\n\n"
            "## Filing a Claim\n\n"
            "### Before You File\n\n"
            "- Check warranty status online\n- Gather proof of purchase\n\n"
            "### How to File\n\n"
            "1. Visit our support portal\n2. Submit claim form\n3. Ship item\n\n"
            "### Claim Resolution\n\n"
            "Claims are processed within 5-7 business days.\n\n"
            "## Troubleshooting\n\n"
            "### Device Won't Turn On\n\n"
            "1. Check power cable\n2. Hold power 10 seconds\n3. Try different outlet\n\n"
            "### Slow Performance\n\n"
            "1. Close unused apps\n2. Restart device\n3. Check storage space\n\n"
            "### Wi-Fi Issues\n\n"
            "1. Restart router\n2. Forget and reconnect\n3. Check for interference\n\n"
            "### Battery Drain\n\n"
            "1. Reduce brightness\n2. Disable background apps\n3. Update firmware\n\n"
            "## Contact Support\n\n"
            "- **Phone:** 1-800-555-0199\n- **Email:** support@novamart.com\n"
            "- **Hours:** Mon-Fri 8am-8pm EST\n"
        )

        chunks = service._chunk_markdown(md, "warranty-doc", "warranty.md")

        # This document has ~20 heading sections but only ~600 tokens total.
        # Must produce far fewer chunks than sections.
        assert len(chunks) < 10, f"Expected <10 chunks, got {len(chunks)}"

        # No chunk should be trivially small (the whole point of the fix)
        for chunk in chunks:
            tokens = len(enc.encode(chunk.content))
            assert tokens >= 30, (
                f"Chunk {chunk.index} is too small: {tokens} tokens"
            )

        # All content should be present across chunks
        combined = "\n".join(c.content for c in chunks)
        assert "Standard Warranty" in combined
        assert "Extended Warranty" in combined
        assert "Filing a Claim" in combined
        assert "Troubleshooting" in combined
        assert "Contact Support" in combined
        assert "1-800-555-0199" in combined


class TestSectionMergingTokenAccuracy:
    """Tests that verify token-level correctness of the merge algorithm."""

    @pytest.fixture
    def service(self) -> IngestionService:
        return IngestionService(
            document_repo=AsyncMock(),
            embedding_service=AsyncMock(),
            vector_store=AsyncMock(),
            chunk_size=100,
            chunk_overlap=15,
        )

    def _make_section(
        self, content: str, heading_path: str = ""
    ) -> MagicMock:
        section = MagicMock()
        section.content = content
        section.heading_path = heading_path
        return section

    def _count_tokens(self, text: str) -> int:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))

    def test_merged_text_token_count_matches_prediction(
        self, service: IngestionService
    ) -> None:
        """The actual token count of joined text must match the merge algorithm's
        internal accounting. This catches separator-counting bugs."""
        sections = [
            self._make_section("Alpha section content here.", "A"),
            self._make_section("Beta section content here.", "B"),
            self._make_section("Gamma section content here.", "C"),
        ]
        groups = service._merge_small_sections(sections)

        for text, _ in groups:
            actual_tokens = self._count_tokens(text)
            assert actual_tokens <= 100, (
                f"Group has {actual_tokens} tokens, exceeds chunk_size=100"
            )

    def test_exact_boundary_two_sections_fit(
        self, service: IngestionService
    ) -> None:
        """Two sections whose combined tokens (with separator) EXACTLY equal
        chunk_size should merge into one group."""
        # Build sections whose total = exactly 100 tokens
        # First, find a text that's ~49 tokens (leaving room for separator)
        base = "word " * 48  # ~48 tokens
        section_a = self._make_section(base.strip(), "A")
        section_b = self._make_section(base.strip(), "B")

        tokens_a = self._count_tokens(section_a.content)
        tokens_b = self._count_tokens(section_b.content)
        sep_tokens = self._count_tokens("\n\n")
        total = tokens_a + sep_tokens + tokens_b

        groups = service._merge_small_sections([section_a, section_b])

        if total <= 100:
            # Should merge into 1 group
            assert len(groups) == 1, (
                f"Expected 1 group for {total} tokens, got {len(groups)}"
            )
        else:
            # Should split into 2 groups
            assert len(groups) == 2

    def test_exact_boundary_one_token_over_splits(
        self, service: IngestionService
    ) -> None:
        """Two sections whose combined tokens are chunk_size + 1 must NOT merge."""
        # Create two sections that together just barely overflow
        # Each is ~51 tokens, so combined ~103 > 100
        text_a = "word " * 50
        text_b = "word " * 50
        section_a = self._make_section(text_a.strip(), "A")
        section_b = self._make_section(text_b.strip(), "B")

        groups = service._merge_small_sections([section_a, section_b])
        assert len(groups) == 2

    def test_no_content_lost_during_merge(
        self, service: IngestionService
    ) -> None:
        """Every section's content must appear in exactly one group."""
        contents = [
            "First section has unique marker AAA.",
            "Second section has unique marker BBB.",
            "Third section has unique marker CCC.",
            "Fourth section has unique marker DDD.",
            "Fifth section has unique marker EEE.",
        ]
        sections = [
            self._make_section(c, f"Path {i}")
            for i, c in enumerate(contents)
        ]
        groups = service._merge_small_sections(sections)

        all_group_text = "\n\n".join(text for text, _ in groups)
        for marker in ["AAA", "BBB", "CCC", "DDD", "EEE"]:
            assert marker in all_group_text, f"Lost content: {marker}"

    def test_deterministic_output(self, service: IngestionService) -> None:
        """Same input sections always produce the same groups."""
        sections = [
            self._make_section(f"Section {i} content.", f"H{i}")
            for i in range(10)
        ]
        groups_1 = service._merge_small_sections(sections)
        groups_2 = service._merge_small_sections(sections)
        assert groups_1 == groups_2


class TestMarkdownIngestionIntegration:
    """Tests that verify _chunk_markdown integrates correctly with process_document."""

    @patch("app.domain.services.ingestion_service.TextExtractor")
    async def test_md_file_uses_markdown_chunker(
        self,
        mock_extractor: MagicMock,
        mock_document_repo: AsyncMock,
        mock_embedding_service: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """A .md file goes through _chunk_markdown, not plain chunking."""
        md_text = (
            "# Policy\n\nIntro text.\n\n"
            "## Section A\n\nContent A.\n\n"
            "## Section B\n\nContent B."
        )
        mock_extractor.extract.return_value = md_text

        doc = Document(
            id="md-doc-1",
            tenant_id="t1",
            filename="policy.md",
            file_type="md",
            chunk_count=0,
            status=DocumentStatus.PENDING,
            uploaded_by="user-1",
        )
        mock_document_repo.update_status.return_value = doc
        mock_embedding_service.embed_batch.return_value = [[0.1]]

        service = IngestionService(
            document_repo=mock_document_repo,
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
        )

        await service.process_document(document=doc, file_content=b"anything")

        # Verify heading_path metadata was passed to vector store
        vector_call = mock_vector_store.add_documents.call_args
        metadatas = vector_call.kwargs["metadatas"]
        assert any(
            "heading_path" in m for m in metadatas
        ), "No heading_path metadata found — markdown chunking not used"

    @patch("app.domain.services.ingestion_service.TextExtractor")
    async def test_md_ingestion_correct_chunk_count(
        self,
        mock_extractor: MagicMock,
        mock_document_repo: AsyncMock,
        mock_embedding_service: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """A small .md file with 3 sections should produce 1 chunk (merged)."""
        md_text = (
            "# Title\n\nOverview.\n\n"
            "## Part 1\n\nShort.\n\n"
            "## Part 2\n\nAlso short."
        )
        mock_extractor.extract.return_value = md_text

        doc = Document(
            id="md-doc-2",
            tenant_id="t1",
            filename="small.md",
            file_type="md",
            chunk_count=0,
            status=DocumentStatus.PENDING,
            uploaded_by="user-1",
        )
        mock_document_repo.update_status.return_value = doc
        mock_embedding_service.embed_batch.return_value = [[0.1]]

        service = IngestionService(
            document_repo=mock_document_repo,
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
        )

        await service.process_document(document=doc, file_content=b"anything")

        # 3 tiny sections merged → 1 chunk → 1 DB record
        assert mock_document_repo.create_chunk.call_count == 1

        # Status updated to READY with chunk_count=1
        ready_call = [
            c for c in mock_document_repo.update_status.call_args_list
            if c.kwargs.get("status") == DocumentStatus.READY
        ]
        assert len(ready_call) == 1
        assert ready_call[0].kwargs["chunk_count"] == 1

    @patch("app.domain.services.ingestion_service.TextExtractor")
    async def test_md_contextualisation_matches_merged_chunk_count(
        self,
        mock_extractor: MagicMock,
        mock_document_repo: AsyncMock,
        mock_embedding_service: AsyncMock,
        mock_vector_store: AsyncMock,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Contextualisation LLM calls should match merged chunk count, not section count."""
        # 10 tiny sections → should merge into ~1-2 chunks
        md_text = "\n\n".join(
            f"## S{i}\n\nContent {i}." for i in range(10)
        )
        mock_extractor.extract.return_value = md_text

        doc = Document(
            id="md-doc-3",
            tenant_id="t1",
            filename="many-sections.md",
            file_type="md",
            chunk_count=0,
            status=DocumentStatus.PENDING,
            uploaded_by="user-1",
        )
        mock_document_repo.update_status.return_value = doc
        mock_embedding_service.embed_batch.return_value = [[0.1]]

        service = IngestionService(
            document_repo=mock_document_repo,
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            llm_provider=mock_llm_provider,
        )

        await service.process_document(document=doc, file_content=b"anything")

        # LLM generate calls should be << 10 (the section count)
        # It should match the merged chunk count
        llm_calls = mock_llm_provider.generate.call_count
        chunk_count = mock_document_repo.create_chunk.call_count
        assert llm_calls == chunk_count, (
            f"LLM calls ({llm_calls}) != chunk count ({chunk_count})"
        )
        assert llm_calls < 10, (
            f"Expected fewer LLM calls than sections (10), got {llm_calls}"
        )

    @patch("app.domain.services.ingestion_service.TextExtractor")
    async def test_txt_file_does_not_use_markdown_chunker(
        self,
        mock_extractor: MagicMock,
        mock_document_repo: AsyncMock,
        mock_embedding_service: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """A .txt file should NOT trigger markdown-aware chunking."""
        text = "# This looks like markdown\n\n## But file_type is txt\n\nContent."
        mock_extractor.extract.return_value = text

        doc = Document(
            id="txt-doc",
            tenant_id="t1",
            filename="notes.txt",
            file_type="txt",
            chunk_count=0,
            status=DocumentStatus.PENDING,
            uploaded_by="user-1",
        )
        mock_document_repo.update_status.return_value = doc
        mock_embedding_service.embed_batch.return_value = [[0.1]]

        service = IngestionService(
            document_repo=mock_document_repo,
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
        )

        await service.process_document(document=doc, file_content=b"anything")

        # No heading_path metadata since it went through plain chunker
        vector_call = mock_vector_store.add_documents.call_args
        metadatas = vector_call.kwargs["metadatas"]
        for m in metadatas:
            assert "heading_path" not in m


class TestMergeWithRealSections:
    """Tests using real MarkdownSection dataclass objects (not mocks).

    Ensures _merge_small_sections works with the actual dataclass attributes,
    not just MagicMock ducktyping.
    """

    @pytest.fixture
    def service(self) -> IngestionService:
        return IngestionService(
            document_repo=AsyncMock(),
            embedding_service=AsyncMock(),
            vector_store=AsyncMock(),
            chunk_size=500,
            chunk_overlap=75,
        )

    def test_real_sections_merge_correctly(
        self, service: IngestionService
    ) -> None:
        """MarkdownSection dataclass objects work with the merge algorithm."""
        from app.rag.markdown_parser import MarkdownSection

        sections = [
            MarkdownSection(content="# Title\n\nIntro.", heading_path="Title", heading_level=1),
            MarkdownSection(content="## A\n\nContent A.", heading_path="Title > A", heading_level=2),
            MarkdownSection(content="## B\n\nContent B.", heading_path="Title > B", heading_level=2),
        ]
        groups = service._merge_small_sections(sections)
        assert len(groups) == 1
        assert "Intro." in groups[0][0]
        assert "Content A." in groups[0][0]
        assert "Content B." in groups[0][0]
        assert groups[0][1] == "Title"

    def test_real_sections_parsed_then_merged(
        self, service: IngestionService
    ) -> None:
        """Full pipeline: parse_markdown_sections → _merge_small_sections."""
        from app.rag.markdown_parser import parse_markdown_sections

        md = "# Root\n\n## A\n\nSmall.\n\n## B\n\nAlso small.\n\n## C\n\nTiny."
        sections = parse_markdown_sections(md)
        groups = service._merge_small_sections(sections)
        # 4 sections, all tiny → should merge into 1 group
        assert len(groups) == 1
        all_text = groups[0][0]
        assert "Small." in all_text
        assert "Also small." in all_text
        assert "Tiny." in all_text

    def test_unicode_content_handled(
        self, service: IngestionService
    ) -> None:
        """Unicode (non-ASCII) content in sections is handled correctly."""
        md = (
            "# Política de Envío\n\n"
            "La política incluye: café, señor, niño.\n\n"
            "## 日本語セクション\n\n"
            "送料ポリシーについて。\n\n"
            "## Раздел на русском\n\n"
            "Политика доставки."
        )
        chunks = service._chunk_markdown(md, "doc-1", "intl.md")
        assert len(chunks) >= 1
        combined = "\n".join(c.content for c in chunks)
        assert "café" in combined
        assert "送料" in combined
        assert "Политика" in combined

    def test_all_heading_only_document(
        self, service: IngestionService
    ) -> None:
        """Document with only headings and no body text still works."""
        md = "# A\n## B\n## C\n### D\n## E\n### F"
        chunks = service._chunk_markdown(md, "doc-1", "headings.md")
        assert len(chunks) >= 1
        combined = "\n".join(c.content for c in chunks)
        assert "# A" in combined
        assert "## E" in combined

    def test_no_content_duplicated_across_groups(
        self, service: IngestionService
    ) -> None:
        """Each section's content appears in exactly one group, never duplicated."""
        from app.rag.markdown_parser import MarkdownSection

        sections = [
            MarkdownSection(
                content=f"## Section {i}\n\nUnique marker SECT{i}END.",
                heading_path=f"Root > Section {i}",
                heading_level=2,
            )
            for i in range(20)
        ]
        groups = service._merge_small_sections(sections)

        for i in range(20):
            marker = f"SECT{i}END"
            count = sum(1 for text, _ in groups if marker in text)
            assert count == 1, (
                f"Marker {marker} found {count} times across groups (expected 1)"
            )

    def test_heading_path_always_from_first_section_in_group(
        self, service: IngestionService
    ) -> None:
        """Verify heading_path invariant: always from the first section in each group."""
        from app.rag.markdown_parser import MarkdownSection

        sections = [
            MarkdownSection(content="A content.", heading_path="Path-A", heading_level=2),
            MarkdownSection(content="B content.", heading_path="Path-B", heading_level=2),
            MarkdownSection(content="C content.", heading_path="Path-C", heading_level=2),
        ]
        groups = service._merge_small_sections(sections)

        # All merge into 1 group, heading must be from first section
        assert len(groups) == 1
        assert groups[0][1] == "Path-A"
        # Verify Path-B and Path-C are NOT the group heading
        assert groups[0][1] != "Path-B"
        assert groups[0][1] != "Path-C"

    def test_group_count_increases_with_document_size(self) -> None:
        """Larger documents produce more groups (sanity check)."""
        from app.rag.markdown_parser import MarkdownSection

        svc = IngestionService(
            document_repo=AsyncMock(),
            embedding_service=AsyncMock(),
            vector_store=AsyncMock(),
            chunk_size=100,
            chunk_overlap=15,
        )

        # 5 sections of ~30 tokens each → ~1-2 groups at chunk_size=100
        small_sections = [
            MarkdownSection(
                content=f"## S{i}\n\nSome content for section number {i}.",
                heading_path=f"S{i}", heading_level=2,
            )
            for i in range(5)
        ]
        # 20 sections of ~30 tokens each → ~6-7 groups at chunk_size=100
        large_sections = [
            MarkdownSection(
                content=f"## S{i}\n\nSome content for section number {i}.",
                heading_path=f"S{i}", heading_level=2,
            )
            for i in range(20)
        ]

        small_groups = svc._merge_small_sections(small_sections)
        large_groups = svc._merge_small_sections(large_sections)
        assert len(large_groups) > len(small_groups)

    def test_merge_preserves_section_ordering(
        self, service: IngestionService
    ) -> None:
        """Sections appear in the same order in groups as in the input."""
        from app.rag.markdown_parser import MarkdownSection

        sections = [
            MarkdownSection(content=f"ORDER_{i}_MARKER", heading_path=f"P{i}", heading_level=2)
            for i in range(10)
        ]
        groups = service._merge_small_sections(sections)

        # Flatten all group text
        all_text = "\n\n".join(text for text, _ in groups)
        # Verify ordering: ORDER_0 appears before ORDER_1, etc.
        for i in range(9):
            pos_i = all_text.index(f"ORDER_{i}_MARKER")
            pos_next = all_text.index(f"ORDER_{i+1}_MARKER")
            assert pos_i < pos_next, (
                f"ORDER_{i} (pos {pos_i}) should appear before ORDER_{i+1} (pos {pos_next})"
            )
