"""Unit tests for ingest API schemas — request/response DTOs.

Tests cover:
- DocumentResponse construction and field types
- DocumentListResponse with empty and populated lists
- DocumentUploadResponse construction
- Field constraint validation
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.api.schemas.ingest import (
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
)
from app.domain.models.enums import DocumentStatus


class TestDocumentResponse:
    """Tests for DocumentResponse schema."""

    def test_valid_construction(self) -> None:
        now = datetime.now(timezone.utc)
        resp = DocumentResponse(
            id="doc-1",
            tenant_id="t-1",
            filename="guide.pdf",
            file_type="pdf",
            chunk_count=10,
            status=DocumentStatus.READY,
            uploaded_by="user-1",
            created_at=now,
        )
        assert resp.id == "doc-1"
        assert resp.filename == "guide.pdf"
        assert resp.status == DocumentStatus.READY
        assert resp.chunk_count == 10

    def test_pending_status(self) -> None:
        resp = DocumentResponse(
            id="doc-2",
            tenant_id="t-1",
            filename="data.csv",
            file_type="csv",
            chunk_count=0,
            status=DocumentStatus.PENDING,
            uploaded_by="user-1",
            created_at=datetime.now(timezone.utc),
        )
        assert resp.status == DocumentStatus.PENDING
        assert resp.chunk_count == 0

    def test_failed_status(self) -> None:
        resp = DocumentResponse(
            id="doc-3",
            tenant_id="t-1",
            filename="broken.pdf",
            file_type="pdf",
            chunk_count=0,
            status=DocumentStatus.FAILED,
            uploaded_by="user-1",
            created_at=datetime.now(timezone.utc),
        )
        assert resp.status == DocumentStatus.FAILED

    def test_missing_required_fields_raises(self) -> None:
        with pytest.raises(ValidationError):
            DocumentResponse()  # type: ignore[call-arg]

    def test_serialization_includes_all_fields(self) -> None:
        now = datetime.now(timezone.utc)
        resp = DocumentResponse(
            id="doc-1",
            tenant_id="t-1",
            filename="test.txt",
            file_type="txt",
            chunk_count=5,
            status=DocumentStatus.PROCESSING,
            uploaded_by="user-1",
            created_at=now,
        )
        data = resp.model_dump()
        assert "id" in data
        assert "tenant_id" in data
        assert "filename" in data
        assert "file_type" in data
        assert "chunk_count" in data
        assert "status" in data
        assert "uploaded_by" in data
        assert "created_at" in data


class TestDocumentListResponse:
    """Tests for DocumentListResponse schema."""

    def test_empty_list(self) -> None:
        resp = DocumentListResponse(documents=[], total=0)
        assert resp.documents == []
        assert resp.total == 0

    def test_populated_list(self) -> None:
        now = datetime.now(timezone.utc)
        docs = [
            DocumentResponse(
                id=f"doc-{i}",
                tenant_id="t-1",
                filename=f"file{i}.txt",
                file_type="txt",
                chunk_count=i,
                status=DocumentStatus.READY,
                uploaded_by="user-1",
                created_at=now,
            )
            for i in range(3)
        ]
        resp = DocumentListResponse(documents=docs, total=3)
        assert len(resp.documents) == 3
        assert resp.total == 3

    def test_total_matches_documents(self) -> None:
        """Total can differ from len(documents) (e.g., pagination)."""
        now = datetime.now(timezone.utc)
        doc = DocumentResponse(
            id="doc-1",
            tenant_id="t-1",
            filename="f.txt",
            file_type="txt",
            chunk_count=0,
            status=DocumentStatus.PENDING,
            uploaded_by="user-1",
            created_at=now,
        )
        resp = DocumentListResponse(documents=[doc], total=100)
        assert resp.total == 100
        assert len(resp.documents) == 1


class TestDocumentUploadResponse:
    """Tests for DocumentUploadResponse schema."""

    def test_valid_construction(self) -> None:
        resp = DocumentUploadResponse(
            document_id="doc-1",
            filename="guide.pdf",
            status="pending",
            message="Document uploaded successfully. Processing will begin shortly.",
        )
        assert resp.document_id == "doc-1"
        assert resp.filename == "guide.pdf"
        assert resp.status == "pending"
        assert "successfully" in resp.message

    def test_missing_required_fields_raises(self) -> None:
        with pytest.raises(ValidationError):
            DocumentUploadResponse()  # type: ignore[call-arg]

    def test_serialization(self) -> None:
        resp = DocumentUploadResponse(
            document_id="doc-1",
            filename="data.csv",
            status="pending",
            message="Uploaded",
        )
        data = resp.model_dump()
        assert data["document_id"] == "doc-1"
        assert data["filename"] == "data.csv"
        assert data["status"] == "pending"
