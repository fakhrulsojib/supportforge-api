"""Document upload and management API router.

Provides CRUD endpoints for document lifecycle management:
- Upload (multipart/form-data) — triggers background ingestion
- List documents for a tenant
- Get document status
- Delete document (admin only)

All endpoints require JWT authentication. Upload and list require
admin or agent role. Delete requires admin role.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends, Request, UploadFile

from app.api.schemas.ingest import (
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
)
from app.core.dependencies import get_embedding_service, get_llm_provider_dep, get_vector_store, require_role
from app.core.exceptions import IngestionError, SupportForgeError
from app.domain.models.enums import UserRole
from app.domain.services.document_service import DocumentService
from app.infrastructure.database.connection import get_async_session
from app.infrastructure.database.repositories.document_repo import SQLDocumentRepository
from app.workers.ingestion_queue import IngestionQueue

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.domain.interfaces.llm_provider import LLMProvider  # noqa: F401
    from app.domain.interfaces.vector_store import VectorStore
    from app.domain.models.user import User
    from app.rag.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


def _get_document_service(session: AsyncSession) -> DocumentService:
    """Wire the DocumentService with its repository dependency."""
    return DocumentService(document_repo=SQLDocumentRepository(session))


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile,
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN, UserRole.AGENT)),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
    llm_provider: LLMProvider | None = Depends(get_llm_provider_dep),
) -> DocumentUploadResponse:
    """Upload a document for RAG ingestion.

    Accepts PDF, Markdown, CSV, and plain text files.
    Maximum file size: 10 MB. Maximum 50 files per tenant.

    The document is created with ``PENDING`` status and background
    ingestion is triggered immediately. The worker processes:
    text extraction → chunking → embedding → vector storage.

    Args:
        file: Uploaded file (multipart/form-data).
        request: FastAPI request (provides access to app state).
        session: Database session.
        user: Authenticated admin or agent user.
        embedding_service: Embedding generation service.
        vector_store: Vector database for storing embeddings.

    Returns:
        DocumentUploadResponse with document ID and status.
    """
    service = _get_document_service(session)

    # Validate file type
    filename = file.filename or ""
    try:
        file_type = service.validate_file_type(filename)
    except IngestionError as exc:
        raise SupportForgeError(
            message=exc.message,
            status_code=415,
            error_code="UNSUPPORTED_FILE_TYPE",
        ) from exc

    # Read file content and validate size
    content = await file.read()
    try:
        service.validate_file_size(len(content))
    except IngestionError as exc:
        if "empty" in exc.message.lower():
            raise SupportForgeError(
                message=exc.message,
                status_code=422,
                error_code="EMPTY_FILE",
            ) from exc
        raise SupportForgeError(
            message=exc.message,
            status_code=413,
            error_code="FILE_TOO_LARGE",
        ) from exc

    # Validate tenant file count
    try:
        await service.validate_file_count(user.tenant_id)
    except IngestionError as exc:
        raise SupportForgeError(
            message=exc.message,
            status_code=422,
            error_code="TENANT_FILE_LIMIT",
        ) from exc

    # Create document record
    document = await service.create_document(
        tenant_id=user.tenant_id,
        filename=filename,
        file_type=file_type,
        uploaded_by=user.id,
    )

    # Commit NOW so the document is immediately visible to list queries.
    # Without this, the document only exists within this session's
    # transaction and won't appear in other sessions (e.g., the list
    # endpoint) until the background task finishes minutes later.
    await session.commit()

    logger.info(
        "document_uploaded",
        document_id=document.id,
        filename=filename,
        file_type=file_type,
        size_bytes=len(content),
        tenant_id=user.tenant_id,
        uploaded_by=user.id,
    )

    # Submit to the bounded-concurrency ingestion queue.
    # The queue uses a semaphore to limit parallel processing,
    # preventing Ollama overload on bulk uploads.
    ingestion_queue: IngestionQueue = request.app.state.ingestion_queue
    await ingestion_queue.submit(
        document_id=document.id,
        file_content=content,
        tenant_id=user.tenant_id,
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_provider=llm_provider,
    )

    return DocumentUploadResponse(
        document_id=document.id,
        filename=filename,
        status="pending",
        message="Document uploaded successfully. Processing will begin shortly.",
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN, UserRole.AGENT)),
) -> DocumentListResponse:
    """List all documents for the authenticated user's tenant.

    Args:
        session: Database session.
        user: Authenticated admin or agent user.

    Returns:
        DocumentListResponse with all tenant documents.
    """
    service = _get_document_service(session)
    documents = await service.list_documents(user.tenant_id)

    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=doc.id,
                tenant_id=doc.tenant_id,
                filename=doc.filename,
                file_type=doc.file_type,
                chunk_count=doc.chunk_count,
                status=doc.status,
                uploaded_by=doc.uploaded_by,
                created_at=doc.created_at,
            )
            for doc in documents
        ],
        total=len(documents),
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document_status(
    document_id: str,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN, UserRole.AGENT)),
) -> DocumentResponse:
    """Get a single document's status and metadata.

    Enforces tenant isolation — users can only access their own
    tenant's documents.

    Args:
        document_id: Document UUID.
        session: Database session.
        user: Authenticated admin or agent user.

    Returns:
        DocumentResponse.

    Raises:
        DocumentNotFoundError: If document doesn't exist or belongs
            to another tenant.
    """
    service = _get_document_service(session)
    document = await service.get_document(document_id, user.tenant_id)

    return DocumentResponse(
        id=document.id,
        tenant_id=document.tenant_id,
        filename=document.filename,
        file_type=document.file_type,
        chunk_count=document.chunk_count,
        status=document.status,
        uploaded_by=document.uploaded_by,
        created_at=document.created_at,
    )


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> None:
    """Delete a document and its chunks (admin only).

    Enforces tenant isolation — admin can only delete their own
    tenant's documents.

    Args:
        document_id: Document UUID.
        session: Database session.
        user: Authenticated admin user.
    """
    service = _get_document_service(session)
    await service.delete_document(document_id, user.tenant_id)

    logger.info(
        "document_deleted",
        document_id=document_id,
        tenant_id=user.tenant_id,
        deleted_by=user.id,
    )
