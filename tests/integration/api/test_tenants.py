"""Integration tests for tenant CRUD API with RBAC enforcement."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.security import create_access_token
from app.domain.models.enums import UserRole
from app.domain.models.tenant import Tenant
from app.domain.models.user import User
from app.main import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# Test JWT secret — must match the .env default
_JWT_SECRET = "change-me-to-another-random-secret"  # noqa: S105


@pytest.fixture
def admin_user() -> User:
    """Admin user fixture."""
    return User(id="admin-1", tenant_id="t-1", email="admin@test.com", role=UserRole.ADMIN)


@pytest.fixture
def viewer_user() -> User:
    """Viewer user fixture."""
    return User(id="viewer-1", tenant_id="t-1", email="viewer@test.com", role=UserRole.VIEWER)


@pytest.fixture
def admin_token() -> str:
    """JWT access token for admin user."""
    return create_access_token(
        user_id="admin-1",
        tenant_id="t-1",
        role="admin",
        secret_key=_JWT_SECRET,
    )


@pytest.fixture
def viewer_token() -> str:
    """JWT access token for viewer user."""
    return create_access_token(
        user_id="viewer-1",
        tenant_id="t-1",
        role="viewer",
        secret_key=_JWT_SECRET,
    )


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async database session."""
    session = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def app_with_mocks(mock_session: AsyncMock, admin_user: User) -> MagicMock:
    """Create app with mocked DB session."""
    app = create_app()

    async def _mock_session_gen() -> AsyncGenerator[AsyncMock, None]:
        yield mock_session

    from app.infrastructure.database.connection import get_async_session

    app.dependency_overrides[get_async_session] = _mock_session_gen
    return app


@pytest.fixture
async def tenant_client(app_with_mocks: MagicMock) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP test client."""
    transport = ASGITransport(app=app_with_mocks)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest.fixture
def sample_tenant() -> Tenant:
    """Sample tenant for testing."""
    return Tenant(id="t-1", name="Acme Corp", slug="acme-corp")


class TestCreateTenantAPI:
    """Tests for POST /api/v1/tenants/."""

    @pytest.mark.asyncio
    async def test_admin_can_create_tenant(
        self,
        tenant_client: AsyncClient,
        admin_token: str,
        admin_user: User,
        sample_tenant: Tenant,
    ) -> None:
        """Admin should be able to create a tenant."""
        with (
            patch("app.api.v1.tenants.SQLTenantRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_slug = AsyncMock(return_value=None)
            mock_repo.create = AsyncMock(return_value=sample_tenant)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await tenant_client.post(
                "/api/v1/tenants",
                json={"name": "Acme Corp", "slug": "acme-corp"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Acme Corp"
        assert data["slug"] == "acme-corp"

    @pytest.mark.asyncio
    async def test_viewer_cannot_create_tenant(
        self,
        tenant_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Viewer should get 401 when trying to create tenant."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await tenant_client.post(
                "/api/v1/tenants",
                json={"name": "Forbidden", "slug": "forbidden"},
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(
        self,
        tenant_client: AsyncClient,
    ) -> None:
        """Missing auth header should return 401 (HTTPBearer rejects)."""
        response = await tenant_client.post(
            "/api/v1/tenants",
            json={"name": "No Auth", "slug": "no-auth"},
        )
        assert response.status_code == 401


class TestGetTenantAPI:
    """Tests for GET /api/v1/tenants/{slug}."""

    @pytest.mark.asyncio
    async def test_any_user_can_get_tenant(
        self,
        tenant_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
        sample_tenant: Tenant,
    ) -> None:
        """Any authenticated user can get tenant by slug."""
        with (
            patch("app.api.v1.tenants.SQLTenantRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_slug = AsyncMock(return_value=sample_tenant)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await tenant_client.get(
                "/api/v1/tenants/acme-corp",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 200
        assert response.json()["slug"] == "acme-corp"

    @pytest.mark.asyncio
    async def test_get_nonexistent_tenant(
        self,
        tenant_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Getting non-existent tenant should return 404."""
        with (
            patch("app.api.v1.tenants.SQLTenantRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_slug = AsyncMock(return_value=None)
            mock_repo.get_by_id = AsyncMock(return_value=None)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await tenant_client.get(
                "/api/v1/tenants/nonexistent",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 404


class TestDeleteTenantAPI:
    """Tests for DELETE /api/v1/tenants/{id}."""

    @pytest.mark.asyncio
    async def test_admin_can_delete(
        self,
        tenant_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Admin should be able to delete a tenant."""
        with (
            patch("app.api.v1.tenants.SQLTenantRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.delete = AsyncMock(return_value=True)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await tenant_client.delete(
                "/api/v1/tenants/t-1",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_viewer_cannot_delete(
        self,
        tenant_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Viewer should not be able to delete a tenant."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await tenant_client.delete(
                "/api/v1/tenants/t-1",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401


class TestListTenantsAPI:
    """Tests for GET /api/v1/tenants/."""

    @pytest.mark.asyncio
    async def test_admin_can_list_tenants(
        self,
        tenant_client: AsyncClient,
        admin_token: str,
        admin_user: User,
        sample_tenant: Tenant,
    ) -> None:
        """Admin should get paginated list of tenants."""
        with (
            patch("app.api.v1.tenants.SQLTenantRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.list_all = AsyncMock(return_value=[sample_tenant])

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await tenant_client.get(
                "/api/v1/tenants",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["tenants"][0]["slug"] == "acme-corp"

    @pytest.mark.asyncio
    async def test_viewer_cannot_list_tenants(
        self,
        tenant_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Viewer should not be able to list tenants (admin only)."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await tenant_client.get(
                "/api/v1/tenants",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_list_empty_tenants(
        self,
        tenant_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Empty tenant list should return total=0."""
        with (
            patch("app.api.v1.tenants.SQLTenantRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.list_all = AsyncMock(return_value=[])

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await tenant_client.get(
                "/api/v1/tenants",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        assert response.json()["total"] == 0
        assert response.json()["tenants"] == []


class TestUpdateTenantAPI:
    """Tests for PATCH /api/v1/tenants/{id}."""

    @pytest.mark.asyncio
    async def test_admin_can_update_tenant(
        self,
        tenant_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Admin should be able to update tenant name."""
        updated_tenant = Tenant(id="t-1", name="Acme Updated", slug="acme-corp")
        with (
            patch("app.api.v1.tenants.SQLTenantRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=updated_tenant)
            mock_repo.update = AsyncMock(return_value=updated_tenant)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await tenant_client.patch(
                "/api/v1/tenants/t-1",
                json={"name": "Acme Updated"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        assert response.json()["name"] == "Acme Updated"

    @pytest.mark.asyncio
    async def test_viewer_cannot_update_tenant(
        self,
        tenant_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Viewer should not be able to update a tenant."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await tenant_client.patch(
                "/api/v1/tenants/t-1",
                json={"name": "Nope"},
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_update_nonexistent_tenant_returns_404(
        self,
        tenant_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Updating non-existent tenant should return 404."""

        with (
            patch("app.api.v1.tenants.SQLTenantRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.update = AsyncMock(return_value=None)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await tenant_client.patch(
                "/api/v1/tenants/nonexistent",
                json={"name": "Nope"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 404


class TestCreateTenantDuplicateSlug:
    """Tests for duplicate slug rejection."""

    @pytest.mark.asyncio
    async def test_duplicate_slug_returns_409(
        self,
        tenant_client: AsyncClient,
        admin_token: str,
        admin_user: User,
        sample_tenant: Tenant,
    ) -> None:
        """Creating tenant with existing slug should return 409."""

        with (
            patch("app.api.v1.tenants.SQLTenantRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_slug = AsyncMock(return_value=sample_tenant)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await tenant_client.post(
                "/api/v1/tenants",
                json={"name": "Acme Corp", "slug": "acme-corp"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 409
