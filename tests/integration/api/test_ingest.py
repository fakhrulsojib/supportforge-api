"""Integration tests for document upload API endpoints.

Tests cover:
- Upload: valid file types (PDF/MD/CSV/TXT) → 201, oversized → 413,
  unsupported type → 415, exceed tenant limit → 422
- List: returns docs for tenant, empty → 200 with total=0
- Get status: existing → 200, non-existent → 404, wrong tenant → 404
- Delete: admin → 204, agent → 401, non-existent → 404, wrong tenant → 404
- Auth: no token → 401, viewer role → 401
- Cross-tenant isolation: Tenant A cannot see/delete Tenant B's docs
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.security import create_access_token
from app.domain.models.document import Document
from app.domain.models.enums import DocumentStatus, UserRole
from app.domain.models.user import User
from app.main import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# Test JWT secret — must match the .env default
_JWT_SECRET = "change-me-to-another-random-secret"  # noqa: S105


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def admin_user() -> User:
    """Admin user in tenant t-1."""
    return User(id="admin-1", tenant_id="t-1", email="admin@test.com", role=UserRole.ADMIN)


@pytest.fixture
def agent_user() -> User:
    """Agent user in tenant t-1."""
    return User(id="agent-1", tenant_id="t-1", email="agent@test.com", role=UserRole.AGENT)


@pytest.fixture
def viewer_user() -> User:
    """Viewer user in tenant t-1."""
    return User(id="viewer-1", tenant_id="t-1", email="viewer@test.com", role=UserRole.VIEWER)


@pytest.fixture
def tenant_b_admin() -> User:
    """Admin user in tenant t-2 (for cross-tenant tests)."""
    return User(id="admin-2", tenant_id="t-2", email="admin2@test.com", role=UserRole.ADMIN)


@pytest.fixture
def admin_token() -> str:
    """JWT for admin in t-1."""
    return create_access_token(user_id="admin-1", tenant_id="t-1", role="admin", secret_key=_JWT_SECRET)


@pytest.fixture
def agent_token() -> str:
    """JWT for agent in t-1."""
    return create_access_token(user_id="agent-1", tenant_id="t-1", role="agent", secret_key=_JWT_SECRET)


@pytest.fixture
def viewer_token() -> str:
    """JWT for viewer in t-1."""
    return create_access_token(user_id="viewer-1", tenant_id="t-1", role="viewer", secret_key=_JWT_SECRET)


@pytest.fixture
def tenant_b_token() -> str:
    """JWT for admin in t-2."""
    return create_access_token(user_id="admin-2", tenant_id="t-2", role="admin", secret_key=_JWT_SECRET)


@pytest.fixture
def mock_session() -> AsyncMock:
    """Mock async database session."""
    session = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def app_with_mocks(mock_session: AsyncMock) -> object:
    """Create app with mocked DB session."""
    app = create_app()

    async def _mock_session_gen() -> AsyncGenerator[AsyncMock, None]:
        yield mock_session

    from app.infrastructure.database.connection import get_async_session

    app.dependency_overrides[get_async_session] = _mock_session_gen
    return app


@pytest.fixture
async def ingest_client(app_with_mocks: object) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP test client."""
    transport = ASGITransport(app=app_with_mocks)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest.fixture
def sample_document() -> Document:
    """Sample document in tenant t-1."""
    return Document(
        id="doc-1",
        tenant_id="t-1",
        filename="guide.pdf",
        file_type="pdf",
        chunk_count=10,
        status=DocumentStatus.READY,
        uploaded_by="admin-1",
    )


def _make_upload_file(content: bytes = b"test content", filename: str = "test.txt") -> dict:
    """Create a file-like dict for multipart upload."""
    return {"file": (filename, io.BytesIO(content), "application/octet-stream")}


# ── Upload Endpoint Tests ────────────────────────────────────────


class TestDocumentUpload:
    """Tests for POST /api/v1/documents/upload."""

    @pytest.mark.asyncio
    async def test_admin_can_upload_txt(
        self,
        ingest_client: AsyncClient,
        admin_token: str,
        admin_user: User,
        sample_document: Document,
    ) -> None:
        """Admin should be able to upload a text file."""
        with (
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
            patch("app.api.v1.ingest.SQLDocumentRepository") as mock_doc_cls,
        ):
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=admin_user)
            mock_doc_repo = mock_doc_cls.return_value
            mock_doc_repo.list_by_tenant = AsyncMock(return_value=[])
            mock_doc_repo.create = AsyncMock(return_value=sample_document)

            response = await ingest_client.post(
                "/api/v1/documents/upload",
                files=_make_upload_file(b"hello world", "notes.txt"),
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 201
        data = response.json()
        assert data["document_id"] == "doc-1"
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_agent_can_upload(
        self,
        ingest_client: AsyncClient,
        agent_token: str,
        agent_user: User,
        sample_document: Document,
    ) -> None:
        """Agent should also be able to upload."""
        with (
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
            patch("app.api.v1.ingest.SQLDocumentRepository") as mock_doc_cls,
        ):
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=agent_user)
            mock_doc_repo = mock_doc_cls.return_value
            mock_doc_repo.list_by_tenant = AsyncMock(return_value=[])
            mock_doc_repo.create = AsyncMock(return_value=sample_document)

            response = await ingest_client.post(
                "/api/v1/documents/upload",
                files=_make_upload_file(b"data", "data.csv"),
                headers={"Authorization": f"Bearer {agent_token}"},
            )

        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_viewer_cannot_upload(
        self,
        ingest_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Viewer should be rejected."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=viewer_user)

            response = await ingest_client.post(
                "/api/v1/documents/upload",
                files=_make_upload_file(b"data", "file.txt"),
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, ingest_client: AsyncClient) -> None:
        """Missing auth header should return 401."""
        response = await ingest_client.post(
            "/api/v1/documents/upload",
            files=_make_upload_file(),
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_unsupported_file_type_returns_415(
        self,
        ingest_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Unsupported file type should return 415."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=admin_user)

            response = await ingest_client.post(
                "/api/v1/documents/upload",
                files=_make_upload_file(b"binary", "image.png"),
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 415

    @pytest.mark.asyncio
    async def test_oversized_file_returns_413(
        self,
        ingest_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """File exceeding 10MB should return 413."""
        oversized = b"x" * (10 * 1024 * 1024 + 1)
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=admin_user)

            response = await ingest_client.post(
                "/api/v1/documents/upload",
                files=_make_upload_file(oversized, "big.txt"),
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 413

    @pytest.mark.asyncio
    async def test_exceed_tenant_file_limit_returns_422(
        self,
        ingest_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Exceeding 50 files per tenant should return 422."""
        existing_docs = [
            Document(filename=f"f{i}.txt", file_type="txt") for i in range(50)
        ]
        with (
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
            patch("app.api.v1.ingest.SQLDocumentRepository") as mock_doc_cls,
        ):
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=admin_user)
            mock_doc_cls.return_value.list_by_tenant = AsyncMock(return_value=existing_docs)

            response = await ingest_client.post(
                "/api/v1/documents/upload",
                files=_make_upload_file(b"data", "new.txt"),
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_upload_pdf(
        self,
        ingest_client: AsyncClient,
        admin_token: str,
        admin_user: User,
        sample_document: Document,
    ) -> None:
        """PDF file type should be accepted."""
        with (
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
            patch("app.api.v1.ingest.SQLDocumentRepository") as mock_doc_cls,
        ):
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=admin_user)
            mock_doc_repo = mock_doc_cls.return_value
            mock_doc_repo.list_by_tenant = AsyncMock(return_value=[])
            mock_doc_repo.create = AsyncMock(return_value=sample_document)

            response = await ingest_client.post(
                "/api/v1/documents/upload",
                files=_make_upload_file(b"%PDF-1.4 test", "report.pdf"),
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_upload_markdown(
        self,
        ingest_client: AsyncClient,
        admin_token: str,
        admin_user: User,
        sample_document: Document,
    ) -> None:
        """Markdown file type should be accepted."""
        with (
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
            patch("app.api.v1.ingest.SQLDocumentRepository") as mock_doc_cls,
        ):
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=admin_user)
            mock_doc_repo = mock_doc_cls.return_value
            mock_doc_repo.list_by_tenant = AsyncMock(return_value=[])
            mock_doc_repo.create = AsyncMock(return_value=sample_document)

            response = await ingest_client.post(
                "/api/v1/documents/upload",
                files=_make_upload_file(b"# Heading", "README.md"),
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_empty_file_returns_422(
        self,
        ingest_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Empty file should be rejected."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=admin_user)

            response = await ingest_client.post(
                "/api/v1/documents/upload",
                files=_make_upload_file(b"", "empty.txt"),
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 422


# ── List Endpoint Tests ──────────────────────────────────────────


class TestDocumentList:
    """Tests for GET /api/v1/documents."""

    @pytest.mark.asyncio
    async def test_admin_can_list_documents(
        self,
        ingest_client: AsyncClient,
        admin_token: str,
        admin_user: User,
        sample_document: Document,
    ) -> None:
        with (
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
            patch("app.api.v1.ingest.SQLDocumentRepository") as mock_doc_cls,
        ):
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=admin_user)
            mock_doc_cls.return_value.list_by_tenant = AsyncMock(return_value=[sample_document])

            response = await ingest_client.get(
                "/api/v1/documents",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["documents"][0]["id"] == "doc-1"

    @pytest.mark.asyncio
    async def test_empty_document_list(
        self,
        ingest_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        with (
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
            patch("app.api.v1.ingest.SQLDocumentRepository") as mock_doc_cls,
        ):
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=admin_user)
            mock_doc_cls.return_value.list_by_tenant = AsyncMock(return_value=[])

            response = await ingest_client.get(
                "/api/v1/documents",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["documents"] == []

    @pytest.mark.asyncio
    async def test_viewer_cannot_list(
        self,
        ingest_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=viewer_user)

            response = await ingest_client.get(
                "/api/v1/documents",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, ingest_client: AsyncClient) -> None:
        response = await ingest_client.get("/api/v1/documents")
        assert response.status_code == 401


# ── Get Status Endpoint Tests ────────────────────────────────────


class TestDocumentGetStatus:
    """Tests for GET /api/v1/documents/{document_id}."""

    @pytest.mark.asyncio
    async def test_get_existing_document(
        self,
        ingest_client: AsyncClient,
        admin_token: str,
        admin_user: User,
        sample_document: Document,
    ) -> None:
        with (
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
            patch("app.api.v1.ingest.SQLDocumentRepository") as mock_doc_cls,
        ):
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=admin_user)
            mock_doc_cls.return_value.get_by_id = AsyncMock(return_value=sample_document)

            response = await ingest_client.get(
                "/api/v1/documents/doc-1",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        assert response.json()["id"] == "doc-1"
        assert response.json()["filename"] == "guide.pdf"

    @pytest.mark.asyncio
    async def test_nonexistent_returns_404(
        self,
        ingest_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        with (
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
            patch("app.api.v1.ingest.SQLDocumentRepository") as mock_doc_cls,
        ):
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=admin_user)
            mock_doc_cls.return_value.get_by_id = AsyncMock(return_value=None)

            response = await ingest_client.get(
                "/api/v1/documents/nonexistent",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_cross_tenant_returns_404(
        self,
        ingest_client: AsyncClient,
        tenant_b_token: str,
        tenant_b_admin: User,
        sample_document: Document,
    ) -> None:
        """Admin from tenant t-2 cannot see document from tenant t-1."""
        with (
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
            patch("app.api.v1.ingest.SQLDocumentRepository") as mock_doc_cls,
        ):
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=tenant_b_admin)
            mock_doc_cls.return_value.get_by_id = AsyncMock(return_value=sample_document)

            response = await ingest_client.get(
                "/api/v1/documents/doc-1",
                headers={"Authorization": f"Bearer {tenant_b_token}"},
            )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_viewer_cannot_get_status(
        self,
        ingest_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=viewer_user)

            response = await ingest_client.get(
                "/api/v1/documents/doc-1",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401


# ── Delete Endpoint Tests ────────────────────────────────────────


class TestDocumentDelete:
    """Tests for DELETE /api/v1/documents/{document_id}."""

    @pytest.mark.asyncio
    async def test_admin_can_delete(
        self,
        ingest_client: AsyncClient,
        admin_token: str,
        admin_user: User,
        sample_document: Document,
    ) -> None:
        with (
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
            patch("app.api.v1.ingest.SQLDocumentRepository") as mock_doc_cls,
        ):
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=admin_user)
            mock_doc_repo = mock_doc_cls.return_value
            mock_doc_repo.get_by_id = AsyncMock(return_value=sample_document)
            mock_doc_repo.delete_chunks_by_document = AsyncMock(return_value=5)
            mock_doc_repo.delete = AsyncMock(return_value=True)

            response = await ingest_client.delete(
                "/api/v1/documents/doc-1",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_agent_cannot_delete(
        self,
        ingest_client: AsyncClient,
        agent_token: str,
        agent_user: User,
    ) -> None:
        """Agent should not be able to delete documents (admin only)."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=agent_user)

            response = await ingest_client.delete(
                "/api/v1/documents/doc-1",
                headers={"Authorization": f"Bearer {agent_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_nonexistent_returns_404(
        self,
        ingest_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        with (
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
            patch("app.api.v1.ingest.SQLDocumentRepository") as mock_doc_cls,
        ):
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=admin_user)
            mock_doc_cls.return_value.get_by_id = AsyncMock(return_value=None)

            response = await ingest_client.delete(
                "/api/v1/documents/nonexistent",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_cross_tenant_delete_returns_404(
        self,
        ingest_client: AsyncClient,
        tenant_b_token: str,
        tenant_b_admin: User,
        sample_document: Document,
    ) -> None:
        """Admin from tenant t-2 cannot delete document from tenant t-1."""
        with (
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
            patch("app.api.v1.ingest.SQLDocumentRepository") as mock_doc_cls,
        ):
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=tenant_b_admin)
            mock_doc_repo = mock_doc_cls.return_value
            mock_doc_repo.get_by_id = AsyncMock(return_value=sample_document)

            response = await ingest_client.delete(
                "/api/v1/documents/doc-1",
                headers={"Authorization": f"Bearer {tenant_b_token}"},
            )

        assert response.status_code == 404
        mock_doc_repo.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_viewer_cannot_delete(
        self,
        ingest_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_cls.return_value.get_by_id = AsyncMock(return_value=viewer_user)

            response = await ingest_client.delete(
                "/api/v1/documents/doc-1",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, ingest_client: AsyncClient) -> None:
        response = await ingest_client.delete("/api/v1/documents/doc-1")
        assert response.status_code == 401
