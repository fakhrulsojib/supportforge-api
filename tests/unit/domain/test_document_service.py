"""Unit tests for DocumentService — domain-layer document management.

Tests cover:
- Happy paths: create, get, list, delete
- Validation: invalid file type, exceed tenant file limit, file size
- Tenant isolation: get/delete with wrong tenant → DocumentNotFoundError
- Edge cases: empty filename, unknown extension, boundary file count
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import DocumentNotFoundError, IngestionError
from app.domain.models.document import Document
from app.domain.models.enums import DocumentStatus
from app.domain.services.document_service import (
    ALLOWED_FILE_TYPES,
    MAX_FILE_SIZE,
    MAX_FILES_PER_TENANT,
    DocumentService,
)


@pytest.fixture
def mock_doc_repo() -> AsyncMock:
    """Create a mock DocumentRepository."""
    repo = AsyncMock()
    repo.create = AsyncMock()
    repo.get_by_id = AsyncMock()
    repo.list_by_tenant = AsyncMock()
    repo.delete = AsyncMock()
    repo.delete_chunks_by_document = AsyncMock()
    return repo


@pytest.fixture
def service(mock_doc_repo: AsyncMock) -> DocumentService:
    """Create a DocumentService with mocked repository."""
    return DocumentService(document_repo=mock_doc_repo)


@pytest.fixture
def sample_document() -> Document:
    """A sample document for testing."""
    return Document(
        id="doc-1",
        tenant_id="t-1",
        filename="guide.pdf",
        file_type="pdf",
        chunk_count=0,
        status=DocumentStatus.PENDING,
        uploaded_by="user-1",
    )


# ── Constants ────────────────────────────────────────────────────


class TestConstants:
    """Verify exported constants have expected values."""

    def test_allowed_file_types(self) -> None:
        assert {"pdf", "md", "csv", "txt"} == ALLOWED_FILE_TYPES

    def test_max_file_size(self) -> None:
        assert MAX_FILE_SIZE == 10 * 1024 * 1024

    def test_max_files_per_tenant(self) -> None:
        assert MAX_FILES_PER_TENANT == 50


# ── validate_file_type ───────────────────────────────────────────


class TestValidateFileType:
    """Tests for DocumentService.validate_file_type()."""

    def test_pdf(self, service: DocumentService) -> None:
        assert service.validate_file_type("report.pdf") == "pdf"

    def test_markdown(self, service: DocumentService) -> None:
        assert service.validate_file_type("README.md") == "md"

    def test_csv(self, service: DocumentService) -> None:
        assert service.validate_file_type("data.csv") == "csv"

    def test_txt(self, service: DocumentService) -> None:
        assert service.validate_file_type("notes.txt") == "txt"

    def test_uppercase_extension(self, service: DocumentService) -> None:
        assert service.validate_file_type("REPORT.PDF") == "pdf"

    def test_mixed_case_extension(self, service: DocumentService) -> None:
        assert service.validate_file_type("data.Csv") == "csv"

    def test_unsupported_type_raises(self, service: DocumentService) -> None:
        with pytest.raises(IngestionError, match="Unsupported file type"):
            service.validate_file_type("image.png")

    def test_no_extension_raises(self, service: DocumentService) -> None:
        with pytest.raises(IngestionError, match="Unsupported file type"):
            service.validate_file_type("noextension")

    def test_empty_filename_raises(self, service: DocumentService) -> None:
        with pytest.raises(IngestionError, match="Filename cannot be empty"):
            service.validate_file_type("")

    def test_dot_only_raises(self, service: DocumentService) -> None:
        with pytest.raises(IngestionError, match="Unsupported file type"):
            service.validate_file_type(".")

    def test_hidden_file_no_ext_raises(self, service: DocumentService) -> None:
        with pytest.raises(IngestionError, match="Unsupported file type"):
            service.validate_file_type(".gitignore")

    def test_double_extension(self, service: DocumentService) -> None:
        """Uses the last extension."""
        assert service.validate_file_type("archive.tar.txt") == "txt"

    def test_exe_rejected(self, service: DocumentService) -> None:
        with pytest.raises(IngestionError, match="Unsupported file type"):
            service.validate_file_type("malware.exe")

    def test_docx_rejected(self, service: DocumentService) -> None:
        with pytest.raises(IngestionError, match="Unsupported file type"):
            service.validate_file_type("document.docx")


# ── validate_file_size ───────────────────────────────────────────


class TestValidateFileSize:
    """Tests for DocumentService.validate_file_size()."""

    def test_small_file_passes(self, service: DocumentService) -> None:
        """1 KB file should pass."""
        service.validate_file_size(1024)

    def test_exactly_max_passes(self, service: DocumentService) -> None:
        """Exactly 10 MB should pass."""
        service.validate_file_size(MAX_FILE_SIZE)

    def test_over_max_raises(self, service: DocumentService) -> None:
        """10 MB + 1 byte should fail."""
        with pytest.raises(IngestionError, match="exceeds maximum"):
            service.validate_file_size(MAX_FILE_SIZE + 1)

    def test_zero_size_raises(self, service: DocumentService) -> None:
        """Empty file should be rejected."""
        with pytest.raises(IngestionError, match="File is empty"):
            service.validate_file_size(0)

    def test_negative_size_raises(self, service: DocumentService) -> None:
        """Negative size should be rejected."""
        with pytest.raises(IngestionError, match="File is empty"):
            service.validate_file_size(-1)


# ── validate_file_count ──────────────────────────────────────────


class TestValidateFileCount:
    """Tests for DocumentService.validate_file_count()."""

    @pytest.mark.asyncio
    async def test_under_limit_passes(
        self, service: DocumentService, mock_doc_repo: AsyncMock
    ) -> None:
        """49 documents should allow one more."""
        mock_doc_repo.list_by_tenant.return_value = [
            Document(filename="f.txt", file_type="txt") for _ in range(49)
        ]
        await service.validate_file_count("t-1")

    @pytest.mark.asyncio
    async def test_at_limit_raises(
        self, service: DocumentService, mock_doc_repo: AsyncMock
    ) -> None:
        """50 documents should reject."""
        mock_doc_repo.list_by_tenant.return_value = [
            Document(filename="f.txt", file_type="txt") for _ in range(50)
        ]
        with pytest.raises(IngestionError, match="Maximum.*50.*documents"):
            await service.validate_file_count("t-1")

    @pytest.mark.asyncio
    async def test_over_limit_raises(
        self, service: DocumentService, mock_doc_repo: AsyncMock
    ) -> None:
        """51 documents should also reject."""
        mock_doc_repo.list_by_tenant.return_value = [
            Document(filename="f.txt", file_type="txt") for _ in range(51)
        ]
        with pytest.raises(IngestionError, match="Maximum.*50.*documents"):
            await service.validate_file_count("t-1")

    @pytest.mark.asyncio
    async def test_empty_tenant_passes(
        self, service: DocumentService, mock_doc_repo: AsyncMock
    ) -> None:
        """Zero documents should allow upload."""
        mock_doc_repo.list_by_tenant.return_value = []
        await service.validate_file_count("t-1")


# ── create_document ──────────────────────────────────────────────


class TestCreateDocument:
    """Tests for DocumentService.create_document()."""

    @pytest.mark.asyncio
    async def test_creates_document_with_pending_status(
        self, service: DocumentService, mock_doc_repo: AsyncMock, sample_document: Document
    ) -> None:
        """Document should be created with PENDING status."""
        mock_doc_repo.create.return_value = sample_document

        result = await service.create_document(
            tenant_id="t-1",
            filename="guide.pdf",
            file_type="pdf",
            uploaded_by="user-1",
        )

        assert result.status == DocumentStatus.PENDING
        assert result.filename == "guide.pdf"
        mock_doc_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_correct_fields_to_repo(
        self, service: DocumentService, mock_doc_repo: AsyncMock, sample_document: Document
    ) -> None:
        """Verify the Document passed to repo has correct fields."""
        mock_doc_repo.create.return_value = sample_document

        await service.create_document(
            tenant_id="t-1",
            filename="data.csv",
            file_type="csv",
            uploaded_by="user-2",
        )

        call_args = mock_doc_repo.create.call_args[0][0]
        assert call_args.tenant_id == "t-1"
        assert call_args.filename == "data.csv"
        assert call_args.file_type == "csv"
        assert call_args.uploaded_by == "user-2"
        assert call_args.status == DocumentStatus.PENDING
        assert call_args.chunk_count == 0


# ── get_document ─────────────────────────────────────────────────


class TestGetDocument:
    """Tests for DocumentService.get_document()."""

    @pytest.mark.asyncio
    async def test_returns_document_when_found(
        self, service: DocumentService, mock_doc_repo: AsyncMock, sample_document: Document
    ) -> None:
        mock_doc_repo.get_by_id.return_value = sample_document

        result = await service.get_document("doc-1", "t-1")

        assert result.id == "doc-1"
        assert result.tenant_id == "t-1"

    @pytest.mark.asyncio
    async def test_not_found_raises(
        self, service: DocumentService, mock_doc_repo: AsyncMock
    ) -> None:
        mock_doc_repo.get_by_id.return_value = None

        with pytest.raises(DocumentNotFoundError):
            await service.get_document("nonexistent", "t-1")

    @pytest.mark.asyncio
    async def test_wrong_tenant_raises(
        self, service: DocumentService, mock_doc_repo: AsyncMock, sample_document: Document
    ) -> None:
        """Document from t-1 should not be accessible by t-2."""
        mock_doc_repo.get_by_id.return_value = sample_document

        with pytest.raises(DocumentNotFoundError):
            await service.get_document("doc-1", "t-2")


# ── list_documents ───────────────────────────────────────────────


class TestListDocuments:
    """Tests for DocumentService.list_documents()."""

    @pytest.mark.asyncio
    async def test_returns_all_tenant_documents(
        self, service: DocumentService, mock_doc_repo: AsyncMock
    ) -> None:
        docs = [
            Document(id=f"doc-{i}", tenant_id="t-1", filename=f"f{i}.txt", file_type="txt")
            for i in range(3)
        ]
        mock_doc_repo.list_by_tenant.return_value = docs

        result = await service.list_documents("t-1")

        assert len(result) == 3
        mock_doc_repo.list_by_tenant.assert_called_once_with("t-1")

    @pytest.mark.asyncio
    async def test_empty_list(
        self, service: DocumentService, mock_doc_repo: AsyncMock
    ) -> None:
        mock_doc_repo.list_by_tenant.return_value = []

        result = await service.list_documents("t-1")

        assert result == []


# ── delete_document ──────────────────────────────────────────────


class TestDeleteDocument:
    """Tests for DocumentService.delete_document()."""

    @pytest.mark.asyncio
    async def test_deletes_existing_document(
        self, service: DocumentService, mock_doc_repo: AsyncMock, sample_document: Document
    ) -> None:
        mock_doc_repo.get_by_id.return_value = sample_document
        mock_doc_repo.delete_chunks_by_document.return_value = 5
        mock_doc_repo.delete.return_value = True

        await service.delete_document("doc-1", "t-1")

        mock_doc_repo.delete_chunks_by_document.assert_called_once_with("doc-1")
        mock_doc_repo.delete.assert_called_once_with("doc-1")

    @pytest.mark.asyncio
    async def test_not_found_raises(
        self, service: DocumentService, mock_doc_repo: AsyncMock
    ) -> None:
        mock_doc_repo.get_by_id.return_value = None

        with pytest.raises(DocumentNotFoundError):
            await service.delete_document("nonexistent", "t-1")

    @pytest.mark.asyncio
    async def test_wrong_tenant_raises(
        self, service: DocumentService, mock_doc_repo: AsyncMock, sample_document: Document
    ) -> None:
        """Cannot delete a document belonging to another tenant."""
        mock_doc_repo.get_by_id.return_value = sample_document

        with pytest.raises(DocumentNotFoundError):
            await service.delete_document("doc-1", "t-2")

        mock_doc_repo.delete.assert_not_called()
