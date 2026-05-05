"""Tests for dependency injection functions."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from starlette.datastructures import State

from app.config import get_settings
from app.core.dependencies import get_app_settings, get_cache, get_current_user, get_tenant_id
from app.core.exceptions import AuthError, TenantNotFoundError
from app.core.security import create_access_token
from app.domain.models.enums import UserRole
from app.domain.models.user import User

# ── Test JWT secret — must match .env default ──
_JWT_SECRET = "change-me-to-another-random-secret"  # noqa: S105


class FakeRequest:
    """Minimal request-like object for testing."""

    def __init__(self, tenant_id: str | None = None) -> None:
        self.state = State()
        if tenant_id is not None:
            self.state.tenant_id = tenant_id


class FakeApp:
    """Minimal app-like object for testing get_cache."""

    def __init__(self, cache: object = None) -> None:
        self.state = State()
        if cache is not None:
            self.state.cache = cache


class FakeRequestWithApp:
    """Request with app attribute for get_cache tests."""

    def __init__(self, app: object) -> None:
        self.app = app


class TestGetTenantId:
    """Test suite for tenant ID extraction."""

    def test_missing_tenant_id_raises_error(self) -> None:
        """Missing tenant_id in state should raise TenantNotFoundError."""
        request = FakeRequest()
        with pytest.raises(TenantNotFoundError, match="Missing X-Tenant-ID"):
            get_tenant_id(request)  # type: ignore[arg-type]

    def test_none_tenant_id_raises_error(self) -> None:
        """None tenant_id in state should raise TenantNotFoundError."""
        request = FakeRequest(tenant_id=None)
        # tenant_id not set on state at all
        with pytest.raises(TenantNotFoundError, match="Missing X-Tenant-ID"):
            get_tenant_id(request)  # type: ignore[arg-type]

    def test_empty_tenant_id_raises_error(self) -> None:
        """Empty string tenant_id should raise TenantNotFoundError."""
        request = FakeRequest()
        request.state.tenant_id = ""
        with pytest.raises(TenantNotFoundError, match="Missing X-Tenant-ID"):
            get_tenant_id(request)  # type: ignore[arg-type]

    def test_valid_tenant_id_returns_id(self) -> None:
        """Valid tenant_id should be returned."""
        request = FakeRequest()
        request.state.tenant_id = "tenant-abc"
        result = get_tenant_id(request)  # type: ignore[arg-type]
        assert result == "tenant-abc"


class TestGetAppSettings:
    """Test suite for settings dependency."""

    def test_returns_settings_instance(self) -> None:
        """get_app_settings should return a Settings object."""
        settings = get_app_settings()
        assert settings.app_name == "SupportForge"


class TestGetCurrentUser:
    """Test suite for JWT-based user extraction."""

    @pytest.mark.asyncio
    async def test_invalid_auth_prefix_raises(self) -> None:
        """Non-Bearer prefix should raise AuthError."""
        mock_session = AsyncMock()
        settings = get_settings()

        with pytest.raises(AuthError, match="Authorization header must be"):
            await get_current_user(
                authorization="Basic dXNlcjpwYXNz",
                session=mock_session,
                settings=settings,
            )

    @pytest.mark.asyncio
    async def test_empty_token_after_bearer_raises(self) -> None:
        """'Bearer ' with no token should raise AuthError."""
        mock_session = AsyncMock()
        settings = get_settings()

        with pytest.raises(AuthError, match="Missing access token"):
            await get_current_user(
                authorization="Bearer ",
                session=mock_session,
                settings=settings,
            )

    @pytest.mark.asyncio
    async def test_user_not_found_raises(self) -> None:
        """Valid JWT but user deleted from DB should raise AuthError."""
        settings = get_settings()
        token = create_access_token(
            user_id="deleted-user",
            tenant_id="t-1",
            role="viewer",
            secret_key=settings.jwt_secret_key,
        )

        mock_session = AsyncMock()

        with patch("app.core.dependencies.SQLUserRepository") as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=None)

            with pytest.raises(AuthError, match="User not found"):
                await get_current_user(
                    authorization=f"Bearer {token}",
                    session=mock_session,
                    settings=settings,
                )

    @pytest.mark.asyncio
    async def test_valid_token_returns_user(self) -> None:
        """Valid JWT + existing user should return User domain model."""
        settings = get_settings()
        token = create_access_token(
            user_id="u-1",
            tenant_id="t-1",
            role="admin",
            secret_key=settings.jwt_secret_key,
        )

        expected_user = User(
            id="u-1",
            tenant_id="t-1",
            email="admin@test.com",
            role=UserRole.ADMIN,
        )
        mock_session = AsyncMock()

        with patch("app.core.dependencies.SQLUserRepository") as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=expected_user)

            user = await get_current_user(
                authorization=f"Bearer {token}",
                session=mock_session,
                settings=settings,
            )

        assert user.id == "u-1"
        assert user.role == UserRole.ADMIN

    @pytest.mark.asyncio
    async def test_expired_token_raises(self) -> None:
        """Expired JWT should raise AuthError."""
        settings = get_settings()
        token = create_access_token(
            user_id="u-1",
            tenant_id="t-1",
            role="viewer",
            secret_key=settings.jwt_secret_key,
            expires_minutes=-1,  # Already expired
        )

        mock_session = AsyncMock()

        with pytest.raises(AuthError):
            await get_current_user(
                authorization=f"Bearer {token}",
                session=mock_session,
                settings=settings,
            )


class TestGetCache:
    """Test suite for cache dependency extraction."""

    def test_returns_cache_when_present(self) -> None:
        """Should return the cache adapter from app.state."""
        fake_cache = object()
        app = FakeApp(cache=fake_cache)
        request = FakeRequestWithApp(app=app)

        result = get_cache(request)  # type: ignore[arg-type]
        assert result is fake_cache

    def test_returns_none_when_no_cache(self) -> None:
        """Should return None when cache is not set on app.state."""
        app = FakeApp()
        request = FakeRequestWithApp(app=app)

        result = get_cache(request)  # type: ignore[arg-type]
        assert result is None
