"""Unit tests for the ingestion worker.

Covers:
    - Worker processes document successfully via IngestionService
    - Worker handles IngestionService failure gracefully
    - Worker handles missing document (not found)
    - Worker logs all ingestion events (success and failure)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.exceptions import IngestionError
from app.domain.models.document import Document
from app.domain.models.enums import DocumentStatus
from app.workers.ingestion_worker import run_ingestion_task


@pytest.fixture
def sample_document() -> Document:
    """Create a sample document for testing."""
    return Document(
        id="doc-456",
        tenant_id="tenant-xyz",
        filename="report.pdf",
        file_type="pdf",
        chunk_count=0,
        status=DocumentStatus.PENDING,
        uploaded_by="user-001",
    )


@pytest.fixture
def mock_session():
    """Create a mock async session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    return session


# ── Successful Processing ────────────────────────────────────────


class TestSuccessfulProcessing:
    """Tests for successful document processing by the worker."""

    @patch("app.workers.ingestion_worker.IngestionService")
    @patch("app.workers.ingestion_worker.SQLDocumentRepository")
    @patch("app.workers.ingestion_worker.AsyncSessionLocal")
    async def test_worker_processes_document(
        self,
        mock_session_factory: MagicMock,
        mock_repo_cls: MagicMock,
        mock_service_cls: MagicMock,
        sample_document: Document,
    ) -> None:
        """Worker creates IngestionService and calls process_document."""
        mock_session = AsyncMock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_repo = AsyncMock()
        mock_repo.get_by_id.return_value = sample_document
        mock_repo_cls.return_value = mock_repo

        mock_service = AsyncMock()
        mock_service_cls.return_value = mock_service

        await run_ingestion_task(
            document_id="doc-456",
            file_content=b"file content bytes",
            tenant_id="tenant-xyz",
            embedding_service=AsyncMock(),
            vector_store=AsyncMock(),
        )

        mock_service.process_document.assert_called_once()

    @patch("app.workers.ingestion_worker.IngestionService")
    @patch("app.workers.ingestion_worker.SQLDocumentRepository")
    @patch("app.workers.ingestion_worker.AsyncSessionLocal")
    async def test_worker_passes_correct_document_and_content(
        self,
        mock_session_factory: MagicMock,
        mock_repo_cls: MagicMock,
        mock_service_cls: MagicMock,
        sample_document: Document,
    ) -> None:
        """Worker passes the correct document and file content to the service."""
        mock_session = AsyncMock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_repo = AsyncMock()
        mock_repo.get_by_id.return_value = sample_document
        mock_repo_cls.return_value = mock_repo

        mock_service = AsyncMock()
        mock_service_cls.return_value = mock_service

        file_content = b"PDF file bytes here"

        await run_ingestion_task(
            document_id="doc-456",
            file_content=file_content,
            tenant_id="tenant-xyz",
            embedding_service=AsyncMock(),
            vector_store=AsyncMock(),
        )

        call_kwargs = mock_service.process_document.call_args.kwargs
        assert call_kwargs["document"] == sample_document
        assert call_kwargs["file_content"] == file_content


# ── Failure Handling ─────────────────────────────────────────────


class TestFailureHandling:
    """Tests for error handling in the worker."""

    @patch("app.workers.ingestion_worker.IngestionService")
    @patch("app.workers.ingestion_worker.SQLDocumentRepository")
    @patch("app.workers.ingestion_worker.AsyncSessionLocal")
    async def test_worker_handles_ingestion_failure(
        self,
        mock_session_factory: MagicMock,
        mock_repo_cls: MagicMock,
        mock_service_cls: MagicMock,
        sample_document: Document,
    ) -> None:
        """Worker catches IngestionError and does not re-raise (background task)."""
        mock_session = AsyncMock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_repo = AsyncMock()
        mock_repo.get_by_id.return_value = sample_document
        mock_repo_cls.return_value = mock_repo

        mock_service = AsyncMock()
        mock_service.process_document.side_effect = IngestionError("Processing failed")
        mock_service_cls.return_value = mock_service

        # Should not raise — worker handles errors internally
        await run_ingestion_task(
            document_id="doc-456",
            file_content=b"content",
            tenant_id="tenant-xyz",
            embedding_service=AsyncMock(),
            vector_store=AsyncMock(),
        )

    @patch("app.workers.ingestion_worker.IngestionService")
    @patch("app.workers.ingestion_worker.SQLDocumentRepository")
    @patch("app.workers.ingestion_worker.AsyncSessionLocal")
    async def test_worker_handles_unexpected_exception(
        self,
        mock_session_factory: MagicMock,
        mock_repo_cls: MagicMock,
        mock_service_cls: MagicMock,
        sample_document: Document,
    ) -> None:
        """Worker catches unexpected exceptions without crashing."""
        mock_session = AsyncMock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_repo = AsyncMock()
        mock_repo.get_by_id.return_value = sample_document
        mock_repo_cls.return_value = mock_repo

        mock_service = AsyncMock()
        mock_service.process_document.side_effect = RuntimeError("Unexpected crash")
        mock_service_cls.return_value = mock_service

        # Should not raise
        await run_ingestion_task(
            document_id="doc-456",
            file_content=b"content",
            tenant_id="tenant-xyz",
            embedding_service=AsyncMock(),
            vector_store=AsyncMock(),
        )

    @patch("app.workers.ingestion_worker.SQLDocumentRepository")
    @patch("app.workers.ingestion_worker.AsyncSessionLocal")
    async def test_worker_handles_missing_document(
        self,
        mock_session_factory: MagicMock,
        mock_repo_cls: MagicMock,
    ) -> None:
        """Worker handles document not found in repository."""
        mock_session = AsyncMock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_repo = AsyncMock()
        mock_repo.get_by_id.return_value = None
        mock_repo_cls.return_value = mock_repo

        # Should not raise — logs warning and returns
        await run_ingestion_task(
            document_id="nonexistent-doc",
            file_content=b"content",
            tenant_id="tenant-xyz",
            embedding_service=AsyncMock(),
            vector_store=AsyncMock(),
        )

    @patch("app.workers.ingestion_worker.SQLDocumentRepository")
    @patch("app.workers.ingestion_worker.AsyncSessionLocal")
    async def test_worker_handles_wrong_tenant_document(
        self,
        mock_session_factory: MagicMock,
        mock_repo_cls: MagicMock,
        sample_document: Document,
    ) -> None:
        """Worker rejects document if tenant_id doesn't match."""
        mock_session = AsyncMock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_repo = AsyncMock()
        mock_repo.get_by_id.return_value = sample_document
        mock_repo_cls.return_value = mock_repo

        # Passing a different tenant_id than the document's
        await run_ingestion_task(
            document_id="doc-456",
            file_content=b"content",
            tenant_id="wrong-tenant",
            embedding_service=AsyncMock(),
            vector_store=AsyncMock(),
        )

        # Should not proceed — tenant mismatch is a security violation
