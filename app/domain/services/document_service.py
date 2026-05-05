"""Domain service for document management.

Pure business logic — NO framework imports. Orchestrates validation
and persistence of uploaded documents through the DocumentRepository port.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.exceptions import DocumentNotFoundError, IngestionError
from app.domain.models.document import Document
from app.domain.models.enums import DocumentStatus

if TYPE_CHECKING:
    from app.domain.interfaces.repository import DocumentRepository

# ── Constants ────────────────────────────────────────────────────

ALLOWED_FILE_TYPES: set[str] = {"pdf", "md", "csv", "txt"}
"""Supported file extensions for document upload."""

MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
"""Maximum allowed file size in bytes."""

MAX_FILES_PER_TENANT: int = 50
"""Maximum number of documents per tenant."""


class DocumentService:
    """Domain service for document lifecycle management.

    Responsibilities:
        - File type and size validation
        - Tenant file count enforcement
        - Document CRUD orchestration with tenant isolation
        - Chunk cleanup on delete

    All persistence is delegated to the injected DocumentRepository.
    """

    def __init__(self, document_repo: DocumentRepository) -> None:
        self._repo = document_repo

    # ── Validation ───────────────────────────────────────────────

    def validate_file_type(self, filename: str) -> str:
        """Validate and extract the file type from a filename.

        Args:
            filename: Original filename with extension.

        Returns:
            Lowercase file extension (e.g., "pdf", "txt").

        Raises:
            IngestionError: If filename is empty or extension is not supported.
        """
        if not filename or not filename.strip():
            raise IngestionError("Filename cannot be empty")

        # Extract extension from the last dot
        parts = filename.rsplit(".", maxsplit=1)
        if len(parts) < 2 or not parts[1]:
            raise IngestionError(f"Unsupported file type. Allowed types: {', '.join(sorted(ALLOWED_FILE_TYPES))}")

        ext = parts[1].lower()
        if ext not in ALLOWED_FILE_TYPES:
            raise IngestionError(
                f"Unsupported file type '.{ext}'. Allowed types: {', '.join(sorted(ALLOWED_FILE_TYPES))}"
            )

        return ext

    def validate_file_size(self, size_bytes: int) -> None:
        """Validate that a file does not exceed the size limit.

        Args:
            size_bytes: File size in bytes.

        Raises:
            IngestionError: If file is empty or exceeds MAX_FILE_SIZE.
        """
        if size_bytes <= 0:
            raise IngestionError("File is empty")

        if size_bytes > MAX_FILE_SIZE:
            max_mb = MAX_FILE_SIZE / (1024 * 1024)
            raise IngestionError(f"File size ({size_bytes:,} bytes) exceeds maximum allowed size ({max_mb:.0f} MB)")

    async def validate_file_count(self, tenant_id: str) -> None:
        """Validate that a tenant has not exceeded the document limit.

        Args:
            tenant_id: Tenant identifier.

        Raises:
            IngestionError: If tenant has reached MAX_FILES_PER_TENANT.
        """
        existing = await self._repo.list_by_tenant(tenant_id)
        if len(existing) >= MAX_FILES_PER_TENANT:
            raise IngestionError(
                f"Maximum of {MAX_FILES_PER_TENANT} documents per tenant reached. "
                f"Delete existing documents before uploading new ones."
            )

    # ── CRUD ─────────────────────────────────────────────────────

    async def create_document(
        self,
        tenant_id: str,
        filename: str,
        file_type: str,
        uploaded_by: str,
    ) -> Document:
        """Create a new document record with PENDING status.

        Args:
            tenant_id: Owning tenant.
            filename: Original filename.
            file_type: Validated file extension.
            uploaded_by: User ID of the uploader.

        Returns:
            The newly created Document.
        """
        document = Document(
            tenant_id=tenant_id,
            filename=filename,
            file_type=file_type,
            chunk_count=0,
            status=DocumentStatus.PENDING,
            uploaded_by=uploaded_by,
        )
        return await self._repo.create(document)

    async def get_document(self, document_id: str, tenant_id: str) -> Document:
        """Get a document by ID with tenant isolation.

        Args:
            document_id: Document UUID.
            tenant_id: Requesting user's tenant ID.

        Returns:
            The Document if found and owned by the tenant.

        Raises:
            DocumentNotFoundError: If not found or belongs to another tenant.
        """
        document = await self._repo.get_by_id(document_id)
        if not document or document.tenant_id != tenant_id:
            raise DocumentNotFoundError(document_id)
        return document

    async def list_documents(self, tenant_id: str) -> list[Document]:
        """List all documents for a tenant.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            List of Documents for the tenant.
        """
        return await self._repo.list_by_tenant(tenant_id)

    async def delete_document(self, document_id: str, tenant_id: str) -> None:
        """Delete a document and its chunks with tenant isolation.

        Deletes chunks first, then the document record.

        Args:
            document_id: Document UUID.
            tenant_id: Requesting user's tenant ID.

        Raises:
            DocumentNotFoundError: If not found or belongs to another tenant.
        """
        document = await self._repo.get_by_id(document_id)
        if not document or document.tenant_id != tenant_id:
            raise DocumentNotFoundError(document_id)

        # Delete chunks first (FK cascade would handle this, but explicit
        # cleanup is preferred for observability and vector store sync)
        await self._repo.delete_chunks_by_document(document_id)
        await self._repo.delete(document_id)
