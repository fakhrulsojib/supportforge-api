"""Domain model for documents and document chunks.

Pure Pydantic models — NO framework imports.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.models.enums import DocumentStatus


class Document(BaseModel):
    """An uploaded document for RAG knowledge base."""

    id: str = ""
    tenant_id: str = ""
    filename: str = Field(..., min_length=1, max_length=500)
    file_type: str = Field(..., min_length=1, max_length=50)
    chunk_count: int = 0
    status: DocumentStatus = DocumentStatus.PENDING
    uploaded_by: str = ""
    created_at: datetime | None = None


class DocumentChunk(BaseModel):
    """A chunk of a document stored in the vector database."""

    id: str = ""
    document_id: str = ""
    chunk_index: int = 0
    content: str = ""
    chroma_id: str = ""
