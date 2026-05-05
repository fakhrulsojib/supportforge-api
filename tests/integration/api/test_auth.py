"""Integration tests for authentication endpoints.

These tests use the FastAPI test client with mocked database
dependencies to test the full auth flow without a real DB.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.security import hash_password
from app.domain.models.enums import UserRole
from app.domain.models.user import User
from app.main import create_app


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async database session."""
    session = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def test_tenant() -> MagicMock:
    """Create a test tenant domain model."""
    from app.domain.models.tenant import Tenant

    return Tenant(id="tenant-1", name="Test Tenant", slug="test-tenant")


@pytest.fixture
def test_user() -> User:
    """Create a test user with hashed password."""
    return User(
        id="user-1",
        tenant_id="tenant-1",
        email="test@example.com",
        password_hash=hash_password("TestPass1!"),
        role=UserRole.VIEWER,
    )


@pytest.fixture
def app_with_mocks(mock_session: AsyncMock) -> MagicMock:
    """Create an app instance with mocked DB session dependency."""
    app = create_app()

    async def _mock_session_gen() -> AsyncGenerator[AsyncMock, None]:
        yield mock_session

    # Override the DB session dependency
    from app.infrastructure.database.connection import get_async_session

    app.dependency_overrides[get_async_session] = _mock_session_gen
    return app


@pytest.fixture
async def auth_client(app_with_mocks: MagicMock) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP test client with mocked dependencies."""
    transport = ASGITransport(app=app_with_mocks)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


class TestRegister:
    """Tests for POST /api/v1/auth/register."""

    @pytest.mark.asyncio
    async def test_register_success(
        self,
        auth_client: AsyncClient,
        mock_session: AsyncMock,
        test_tenant: MagicMock,
    ) -> None:
        """Successful registration should return 201 with tokens."""
        with (
            patch("app.api.v1.auth.SQLTenantRepository") as mock_tenant_cls,
            patch("app.api.v1.auth.SQLUserRepository") as mock_user_cls,
        ):
            mock_tenant_repo = mock_tenant_cls.return_value
            mock_tenant_repo.get_by_id = AsyncMock(return_value=test_tenant)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_email = AsyncMock(return_value=None)
            mock_user_repo.create = AsyncMock(
                return_value=User(
                    id="new-user-1",
                    tenant_id="tenant-1",
                    email="new@example.com",
                    role=UserRole.VIEWER,
                )
            )

            # Mock the direct session.get for password hash update
            mock_session.get = AsyncMock(return_value=MagicMock())

            response = await auth_client.post(
                "/api/v1/auth/register",
                json={
                    "email": "new@example.com",
                    "password": "StrongP@ss1",
                    "tenant_id": "tenant-1",
                },
            )

        assert response.status_code == 201
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    @pytest.mark.asyncio
    async def test_register_tenant_not_found(
        self,
        auth_client: AsyncClient,
    ) -> None:
        """Registration with non-existent tenant should return 404."""
        with patch("app.api.v1.auth.SQLTenantRepository") as mock_tenant_cls:
            mock_tenant_repo = mock_tenant_cls.return_value
            mock_tenant_repo.get_by_id = AsyncMock(return_value=None)

            response = await auth_client.post(
                "/api/v1/auth/register",
                json={
                    "email": "test@example.com",
                    "password": "StrongP@ss1",
                    "tenant_id": "nonexistent",
                },
            )

        assert response.status_code == 404
        assert "TENANT_NOT_FOUND" in response.json()["error"]["code"]

    @pytest.mark.asyncio
    async def test_register_duplicate_email(
        self,
        auth_client: AsyncClient,
        test_tenant: MagicMock,
        test_user: User,
    ) -> None:
        """Registration with existing email should return 409."""
        with (
            patch("app.api.v1.auth.SQLTenantRepository") as mock_tenant_cls,
            patch("app.api.v1.auth.SQLUserRepository") as mock_user_cls,
        ):
            mock_tenant_repo = mock_tenant_cls.return_value
            mock_tenant_repo.get_by_id = AsyncMock(return_value=test_tenant)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_email = AsyncMock(return_value=test_user)

            response = await auth_client.post(
                "/api/v1/auth/register",
                json={
                    "email": "test@example.com",
                    "password": "StrongP@ss1",
                    "tenant_id": "tenant-1",
                },
            )

        assert response.status_code == 409
        assert "EMAIL_ALREADY_EXISTS" in response.json()["error"]["code"]

    @pytest.mark.asyncio
    async def test_register_weak_password(
        self,
        auth_client: AsyncClient,
        test_tenant: MagicMock,
    ) -> None:
        """Registration with weak password should return 422."""
        with (
            patch("app.api.v1.auth.SQLTenantRepository") as mock_tenant_cls,
        ):
            mock_tenant_repo = mock_tenant_cls.return_value
            mock_tenant_repo.get_by_id = AsyncMock(return_value=test_tenant)

            response = await auth_client.post(
                "/api/v1/auth/register",
                json={
                    "email": "test@example.com",
                    "password": "weakpassword",  # No uppercase, digit, or special char
                    "tenant_id": "tenant-1",
                },
            )

        assert response.status_code == 422
        assert "WEAK_PASSWORD" in response.json()["error"]["code"]

    @pytest.mark.asyncio
    async def test_register_invalid_role(
        self,
        auth_client: AsyncClient,
        test_tenant: MagicMock,
    ) -> None:
        """Registration with invalid role should return 422."""
        with patch("app.api.v1.auth.SQLTenantRepository") as mock_tenant_cls:
            mock_tenant_repo = mock_tenant_cls.return_value
            mock_tenant_repo.get_by_id = AsyncMock(return_value=test_tenant)

            response = await auth_client.post(
                "/api/v1/auth/register",
                json={
                    "email": "test@example.com",
                    "password": "StrongP@ss1",
                    "tenant_id": "tenant-1",
                    "role": "superadmin",
                },
            )

        assert response.status_code == 422
        assert "INVALID_ROLE" in response.json()["error"]["code"]


class TestLogin:
    """Tests for POST /api/v1/auth/login."""

    @pytest.mark.asyncio
    async def test_login_success(
        self,
        auth_client: AsyncClient,
        test_user: User,
    ) -> None:
        """Successful login should return tokens."""
        with patch("app.api.v1.auth.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_email = AsyncMock(return_value=test_user)

            response = await auth_client.post(
                "/api/v1/auth/login",
                json={
                    "email": "test@example.com",
                    "password": "TestPass1!",
                    "tenant_id": "tenant-1",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_wrong_password(
        self,
        auth_client: AsyncClient,
        test_user: User,
    ) -> None:
        """Login with wrong password should return 401."""
        with patch("app.api.v1.auth.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_email = AsyncMock(return_value=test_user)

            response = await auth_client.post(
                "/api/v1/auth/login",
                json={
                    "email": "test@example.com",
                    "password": "WrongPass1!",
                    "tenant_id": "tenant-1",
                },
            )

        assert response.status_code == 401
        assert "AUTH_ERROR" in response.json()["error"]["code"]

    @pytest.mark.asyncio
    async def test_login_nonexistent_user(
        self,
        auth_client: AsyncClient,
    ) -> None:
        """Login with non-existent email should return 401."""
        with patch("app.api.v1.auth.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_email = AsyncMock(return_value=None)

            response = await auth_client.post(
                "/api/v1/auth/login",
                json={
                    "email": "nobody@example.com",
                    "password": "TestPass1!",
                    "tenant_id": "tenant-1",
                },
            )

        assert response.status_code == 401


class TestRefresh:
    """Tests for POST /api/v1/auth/refresh."""

    @pytest.mark.asyncio
    async def test_refresh_success(
        self,
        auth_client: AsyncClient,
        test_user: User,
    ) -> None:
        """Valid refresh token should return new access token."""
        from app.core.security import create_refresh_token

        refresh_token = create_refresh_token(
            user_id="user-1",
            tenant_id="tenant-1",
            secret_key="change-me-to-another-random-secret",
        )

        with patch("app.api.v1.auth.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=test_user)

            response = await auth_client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": refresh_token},
            )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["refresh_token"] == refresh_token

    @pytest.mark.asyncio
    async def test_refresh_invalid_token(
        self,
        auth_client: AsyncClient,
    ) -> None:
        """Invalid refresh token should return 401."""
        response = await auth_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "invalid.token.here"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_user_deleted(
        self,
        auth_client: AsyncClient,
    ) -> None:
        """Refresh for deleted user should return 401."""
        from app.core.security import create_refresh_token

        refresh_token = create_refresh_token(
            user_id="deleted-user",
            tenant_id="tenant-1",
            secret_key="change-me-to-another-random-secret",
        )

        with patch("app.api.v1.auth.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=None)

            response = await auth_client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": refresh_token},
            )

        assert response.status_code == 401
