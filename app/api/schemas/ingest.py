"""Ingest API schemas — request/response DTOs for document management.

These schemas define the API contract for document upload, listing,
status retrieval, and deletion endpoints.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003 — Pydantic needs runtime access

from pydantic import BaseModel, Field

from app.domain.models.enums import DocumentStatus  # noqa: TCH001 — Pydantic needs runtime access


class DocumentResponse(BaseModel):
    """Response body for a single document."""

    id: str = Field(..., description="Document UUID")
    tenant_id: str = Field(..., description="Owning tenant ID")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File extension (pdf, md, csv, txt)")
    chunk_count: int = Field(..., description="Number of chunks after processing")
    status: DocumentStatus = Field(..., description="Processing status")
    uploaded_by: str = Field(..., description="User ID of the uploader")
    created_at: datetime | None = Field(None, description="Upload timestamp")


class DocumentListResponse(BaseModel):
    """Response body for listing documents."""

    documents: list[DocumentResponse] = Field(default_factory=list, description="List of documents")
    total: int = Field(..., description="Total number of documents")


class DocumentUploadResponse(BaseModel):
    """Response body for successful document upload."""

    document_id: str = Field(..., description="Newly created document UUID")
    filename: str = Field(..., description="Uploaded filename")
    status: str = Field("pending", description="Initial processing status")
    message: str = Field(..., description="Human-readable status message")
