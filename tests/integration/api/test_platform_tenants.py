"""Integration tests for platform tenant provisioning API (superadmin-only).

Tests cover:
- POST /api/v1/platform/tenants — superadmin creates tenant
- GET /api/v1/platform/tenants — paginated listing with status filter
- PATCH /api/v1/platform/tenants/{id}/status — status transitions
- Auth enforcement: admin/viewer → 401, no auth → 401
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.security import create_access_token
from app.domain.models.enums import TenantStatus, UserRole
from app.domain.models.tenant import Tenant
from app.domain.models.user import User
from app.main import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# Test JWT secret — must match the .env default
_JWT_SECRET = "change-me-to-another-random-secret"  # noqa: S105


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def superadmin_user() -> User:
    """Superadmin user fixture."""
    return User(id="sa-1", tenant_id="t-platform", email="superadmin@platform.com", role=UserRole.SUPERADMIN)


@pytest.fixture
def admin_user() -> User:
    """Admin user fixture (non-superadmin)."""
    return User(id="admin-1", tenant_id="t-1", email="admin@test.com", role=UserRole.ADMIN)


@pytest.fixture
def viewer_user() -> User:
    """Viewer user fixture."""
    return User(id="viewer-1", tenant_id="t-1", email="viewer@test.com", role=UserRole.VIEWER)


@pytest.fixture
def superadmin_token() -> str:
    """JWT access token for superadmin user."""
    return create_access_token(
        user_id="sa-1",
        tenant_id="t-platform",
        role="superadmin",
        secret_key=_JWT_SECRET,
    )


@pytest.fixture
def admin_token() -> str:
    """JWT access token for regular admin user."""
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
def app_with_mocks(mock_session: AsyncMock) -> MagicMock:
    """Create app with mocked DB session."""
    app = create_app()

    async def _mock_session_gen() -> AsyncGenerator[AsyncMock, None]:
        yield mock_session

    from app.infrastructure.database.connection import get_async_session

    app.dependency_overrides[get_async_session] = _mock_session_gen
    return app


@pytest.fixture
async def platform_client(app_with_mocks: MagicMock) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP test client."""
    transport = ASGITransport(app=app_with_mocks)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest.fixture
def sample_tenant() -> Tenant:
    """Sample active tenant."""
    return Tenant(id="t-new", name="New Corp", slug="new-corp", status=TenantStatus.ACTIVE)


# ── POST /api/v1/platform/tenants ──────────────────────────────

class TestPlatformCreateTenant:
    """Tests for POST /api/v1/platform/tenants."""

    @pytest.mark.asyncio
    async def test_superadmin_creates_tenant(
        self,
        platform_client: AsyncClient,
        superadmin_token: str,
        superadmin_user: User,
        sample_tenant: Tenant,
    ) -> None:
        """Superadmin should be able to create a tenant."""
        with (
            patch("app.api.v1.platform_tenants.SQLTenantRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_slug = AsyncMock(return_value=None)
            mock_repo.create = AsyncMock(return_value=sample_tenant)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=superadmin_user)

            response = await platform_client.post(
                "/api/v1/platform/tenants/",
                json={"name": "New Corp", "slug": "new-corp"},
                headers={"Authorization": f"Bearer {superadmin_token}"},
            )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "New Corp"
        assert data["slug"] == "new-corp"
        assert data["status"] == "active"

    @pytest.mark.asyncio
    async def test_admin_cannot_create_platform_tenant(
        self,
        platform_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Regular admin should get 401 for platform tenant creation."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await platform_client.post(
                "/api/v1/platform/tenants/",
                json={"name": "Forbidden", "slug": "forbidden"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_viewer_cannot_create_platform_tenant(
        self,
        platform_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Viewer should get 401 for platform tenant creation."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await platform_client.post(
                "/api/v1/platform/tenants/",
                json={"name": "Nope", "slug": "nope"},
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(
        self,
        platform_client: AsyncClient,
    ) -> None:
        """Missing auth header should return 401."""
        response = await platform_client.post(
            "/api/v1/platform/tenants/",
            json={"name": "No Auth", "slug": "no-auth"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_duplicate_slug_returns_409(
        self,
        platform_client: AsyncClient,
        superadmin_token: str,
        superadmin_user: User,
        sample_tenant: Tenant,
    ) -> None:
        """Creating tenant with existing slug should return 409."""
        with (
            patch("app.api.v1.platform_tenants.SQLTenantRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_slug = AsyncMock(return_value=sample_tenant)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=superadmin_user)

            response = await platform_client.post(
                "/api/v1/platform/tenants/",
                json={"name": "New Corp", "slug": "new-corp"},
                headers={"Authorization": f"Bearer {superadmin_token}"},
            )

        assert response.status_code == 409


# ── GET /api/v1/platform/tenants ────────────────────────────────

class TestPlatformListTenants:
    """Tests for GET /api/v1/platform/tenants."""

    @pytest.mark.asyncio
    async def test_superadmin_can_list(
        self,
        platform_client: AsyncClient,
        superadmin_token: str,
        superadmin_user: User,
        sample_tenant: Tenant,
    ) -> None:
        """Superadmin should get paginated tenant list."""
        with (
            patch("app.api.v1.platform_tenants.SQLTenantRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.list_all_with_status = AsyncMock(return_value=[sample_tenant])
            mock_repo.count_all = AsyncMock(return_value=1)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=superadmin_user)

            response = await platform_client.get(
                "/api/v1/platform/tenants/",
                headers={"Authorization": f"Bearer {superadmin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["tenants"][0]["slug"] == "new-corp"
        assert data["tenants"][0]["status"] == "active"

    @pytest.mark.asyncio
    async def test_list_with_status_filter(
        self,
        platform_client: AsyncClient,
        superadmin_token: str,
        superadmin_user: User,
    ) -> None:
        """Status filter should be passed to repository."""
        suspended_tenant = Tenant(
            id="t-2", name="Suspended Corp", slug="suspended-corp",
            status=TenantStatus.SUSPENDED,
        )
        with (
            patch("app.api.v1.platform_tenants.SQLTenantRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.list_all_with_status = AsyncMock(return_value=[suspended_tenant])
            mock_repo.count_all = AsyncMock(return_value=1)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=superadmin_user)

            response = await platform_client.get(
                "/api/v1/platform/tenants/?status=suspended",
                headers={"Authorization": f"Bearer {superadmin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["tenants"][0]["status"] == "suspended"

    @pytest.mark.asyncio
    async def test_list_empty(
        self,
        platform_client: AsyncClient,
        superadmin_token: str,
        superadmin_user: User,
    ) -> None:
        """Empty result should return total=0."""
        with (
            patch("app.api.v1.platform_tenants.SQLTenantRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.list_all_with_status = AsyncMock(return_value=[])
            mock_repo.count_all = AsyncMock(return_value=0)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=superadmin_user)

            response = await platform_client.get(
                "/api/v1/platform/tenants/",
                headers={"Authorization": f"Bearer {superadmin_token}"},
            )

        assert response.status_code == 200
        assert response.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_admin_cannot_list(
        self,
        platform_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Regular admin should get 401."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await platform_client.get(
                "/api/v1/platform/tenants/",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 401


# ── PATCH /api/v1/platform/tenants/{id}/status ──────────────────

class TestPlatformUpdateTenantStatus:
    """Tests for PATCH /api/v1/platform/tenants/{id}/status."""

    @pytest.mark.asyncio
    async def test_active_to_suspended(
        self,
        platform_client: AsyncClient,
        superadmin_token: str,
        superadmin_user: User,
    ) -> None:
        """Superadmin should transition active → suspended."""
        active_tenant = Tenant(
            id="t-1", name="Test", slug="test", status=TenantStatus.ACTIVE,
        )
        suspended_tenant = Tenant(
            id="t-1", name="Test", slug="test", status=TenantStatus.SUSPENDED,
        )
        with (
            patch("app.api.v1.platform_tenants.SQLTenantRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=active_tenant)
            mock_repo.update_status = AsyncMock(return_value=suspended_tenant)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=superadmin_user)

            response = await platform_client.patch(
                "/api/v1/platform/tenants/t-1/status",
                json={"status": "suspended"},
                headers={"Authorization": f"Bearer {superadmin_token}"},
            )

        assert response.status_code == 200
        assert response.json()["status"] == "suspended"

    @pytest.mark.asyncio
    async def test_archived_is_terminal(
        self,
        platform_client: AsyncClient,
        superadmin_token: str,
        superadmin_user: User,
    ) -> None:
        """Archived → active should return 400."""
        archived_tenant = Tenant(
            id="t-1", name="Test", slug="test", status=TenantStatus.ARCHIVED,
        )
        with (
            patch("app.api.v1.platform_tenants.SQLTenantRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=archived_tenant)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=superadmin_user)

            response = await platform_client.patch(
                "/api/v1/platform/tenants/t-1/status",
                json={"status": "active"},
                headers={"Authorization": f"Bearer {superadmin_token}"},
            )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_not_found_returns_404(
        self,
        platform_client: AsyncClient,
        superadmin_token: str,
        superadmin_user: User,
    ) -> None:
        """Non-existent tenant should return 404."""
        with (
            patch("app.api.v1.platform_tenants.SQLTenantRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=None)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=superadmin_user)

            response = await platform_client.patch(
                "/api/v1/platform/tenants/missing/status",
                json={"status": "suspended"},
                headers={"Authorization": f"Bearer {superadmin_token}"},
            )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_admin_cannot_update_status(
        self,
        platform_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Regular admin should get 401."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await platform_client.patch(
                "/api/v1/platform/tenants/t-1/status",
                json={"status": "suspended"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_status_value(
        self,
        platform_client: AsyncClient,
        superadmin_token: str,
        superadmin_user: User,
    ) -> None:
        """Invalid status value should return 422."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=superadmin_user)

            response = await platform_client.patch(
                "/api/v1/platform/tenants/t-1/status",
                json={"status": "deleted"},
                headers={"Authorization": f"Bearer {superadmin_token}"},
            )

        assert response.status_code == 422


# ── Chat gate enforcement ───────────────────────────────────────

class TestChatGateEnforcement:
    """Tests for tenant status gate on REST chat endpoint."""

    @pytest.mark.asyncio
    async def test_suspended_tenant_chat_rejected(
        self,
        platform_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Suspended tenant should get 403 when trying to chat."""
        suspended_tenant = Tenant(
            id="t-1", name="Test", slug="test", status=TenantStatus.SUSPENDED,
        )
        # Set a mock chat_service so the get_chat_service dependency doesn't raise
        platform_client._transport.app.state.chat_service = AsyncMock()  # type: ignore[union-attr]

        with (
            patch("app.api.v1.chat_router.SQLTenantRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=suspended_tenant)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await platform_client.post(
                "/api/v1/chat",
                json={"message": "Hello"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "TENANT_SUSPENDED"

    @pytest.mark.asyncio
    async def test_archived_tenant_chat_rejected(
        self,
        platform_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Archived tenant should get 403 when trying to chat."""
        archived_tenant = Tenant(
            id="t-1", name="Test", slug="test", status=TenantStatus.ARCHIVED,
        )
        platform_client._transport.app.state.chat_service = AsyncMock()  # type: ignore[union-attr]

        with (
            patch("app.api.v1.chat_router.SQLTenantRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=archived_tenant)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await platform_client.post(
                "/api/v1/chat",
                json={"message": "Hello"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_active_tenant_chat_allowed(
        self,
        platform_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Active tenant should pass through the gate (mock chat service)."""
        active_tenant = Tenant(
            id="t-1", name="Test", slug="test", status=TenantStatus.ACTIVE,
        )
        mock_chat = AsyncMock()
        mock_chat.process_message = AsyncMock(return_value={
            "answer": "Hello!",
            "conversation_id": "conv-1",
            "sources": [],
            "escalated": False,
            "escalation_reason": "",
            "escalation_trigger": "none",
            "model_used": "test-model",
        })
        platform_client._transport.app.state.chat_service = mock_chat  # type: ignore[union-attr]

        with (
            patch("app.api.v1.chat_router.SQLTenantRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=active_tenant)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await platform_client.post(
                "/api/v1/chat",
                json={"message": "Hello"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        assert response.json()["answer"] == "Hello!"


