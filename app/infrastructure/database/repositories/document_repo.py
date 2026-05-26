"""SQLAlchemy implementation of DocumentRepository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import delete, select

from app.domain.interfaces.repository import DocumentRepository
from app.domain.models.document import Document, DocumentChunk
from app.domain.models.enums import DocumentStatus
from app.infrastructure.database.models import DocumentChunkModel, DocumentModel

import structlog

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class SQLDocumentRepository(DocumentRepository):
    """Concrete document repository backed by PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def _to_domain(self, model: DocumentModel) -> Document:
        """Convert ORM model to domain model."""
        return Document(
            id=model.id,
            tenant_id=model.tenant_id,
            filename=model.filename,
            file_type=model.file_type,
            chunk_count=model.chunk_count,
            status=model.status,
            uploaded_by=model.uploaded_by or "",
            created_at=model.created_at,
        )

    def _chunk_to_domain(self, model: DocumentChunkModel) -> DocumentChunk:
        """Convert ORM chunk model to domain model."""
        return DocumentChunk(
            id=model.id,
            document_id=model.document_id,
            chunk_index=model.chunk_index,
            content=model.content,
            chroma_id=model.chroma_id,
        )

    async def create(self, document: Document) -> Document:
        """Create a new document record."""
        logger.debug("repo_create_document", tenant_id=document.tenant_id, filename=document.filename)
        model = DocumentModel(
            tenant_id=document.tenant_id,
            filename=document.filename,
            file_type=document.file_type,
            chunk_count=document.chunk_count,
            status=document.status,
            uploaded_by=document.uploaded_by or None,
        )
        self._session.add(model)
        await self._session.flush()
        return self._to_domain(model)

    async def get_by_id(self, document_id: str) -> Document | None:
        """Get a document by ID."""
        logger.debug("repo_get_document_by_id", document_id=document_id)
        result = await self._session.get(DocumentModel, document_id)
        return self._to_domain(result) if result else None

    async def list_by_tenant(self, tenant_id: str) -> list[Document]:
        """List all documents for a tenant."""
        stmt = (
            select(DocumentModel).where(DocumentModel.tenant_id == tenant_id).order_by(DocumentModel.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return [self._to_domain(m) for m in result.scalars().all()]

    async def update_status(self, document_id: str, status: DocumentStatus, chunk_count: int = 0, *, reset_chunk_count: bool = False) -> Document | None:
        """Update a document's processing status."""
        model = await self._session.get(DocumentModel, document_id)
        if not model:
            return None
        model.status = status
        if chunk_count > 0:
            model.chunk_count = chunk_count
        elif reset_chunk_count:
            model.chunk_count = 0
        await self._session.flush()
        return self._to_domain(model)

    async def delete(self, document_id: str) -> bool:
        """Delete a document and its chunks."""
        logger.debug("repo_delete_document", document_id=document_id)
        model = await self._session.get(DocumentModel, document_id)
        if not model:
            return False
        await self._session.delete(model)
        await self._session.flush()
        return True

    async def create_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Create a document chunk record."""
        model = DocumentChunkModel(
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            chroma_id=chunk.chroma_id,
        )
        self._session.add(model)
        await self._session.flush()
        return self._chunk_to_domain(model)

    async def get_chunks_by_document(self, document_id: str) -> list[DocumentChunk]:
        """Get all chunks for a document ordered by index."""
        stmt = (
            select(DocumentChunkModel)
            .where(DocumentChunkModel.document_id == document_id)
            .order_by(DocumentChunkModel.chunk_index)
        )
        result = await self._session.execute(stmt)
        return [self._chunk_to_domain(m) for m in result.scalars().all()]

    async def delete_chunks_by_document(self, document_id: str) -> int:
        """Delete all chunks for a document. Returns count of deleted chunks."""
        stmt = delete(DocumentChunkModel).where(DocumentChunkModel.document_id == document_id)
        result = await self._session.execute(stmt)
        await self._session.flush()
        return result.rowcount  # type: ignore[return-value]
