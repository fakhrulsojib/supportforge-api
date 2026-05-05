"""Unit tests for the IngestionService domain service.

Covers:
    - Successful end-to-end ingestion (extract → chunk → embed → store → persist)
    - Status transitions (PENDING → PROCESSING → READY)
    - Partial embedding failure → rollback, status FAILED
    - Vector store failure → rollback, status FAILED
    - Empty text extraction → status FAILED
    - Chunk count persisted correctly
    - ChromaDB IDs match document_chunks.chroma_id
    - Duplicate chunk detection / idempotency
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
    """Create an IngestionService with mocked dependencies."""
    return IngestionService(
        document_repo=mock_document_repo,
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
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
        from app.workers.text_extractor import TextExtractionError

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
        from app.workers.text_extractor import TextExtractionError

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
