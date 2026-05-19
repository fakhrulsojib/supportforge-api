"""Integration tests for voice API endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.security import create_access_token
from app.core.tenant_config import TenantVoiceConfig
from app.domain.models.enums import UserRole
from app.domain.models.tenant import Tenant
from app.domain.models.user import User
from app.main import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

_JWT_SECRET = "change-me-to-another-random-secret"  # noqa: S105


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def admin_user() -> User:
    """Authenticated admin user."""
    return User(id="admin-1", tenant_id="t-1", email="admin@test.com", role=UserRole.ADMIN)


@pytest.fixture
def admin_token() -> str:
    """JWT for admin user."""
    return create_access_token(
        user_id="admin-1",
        tenant_id="t-1",
        role="admin",
        secret_key=_JWT_SECRET,
    )


@pytest.fixture
def viewer_user() -> User:
    """Authenticated viewer user."""
    return User(id="viewer-1", tenant_id="t-1", email="viewer@test.com", role=UserRole.VIEWER)


@pytest.fixture
def viewer_token() -> str:
    """JWT for viewer user."""
    return create_access_token(
        user_id="viewer-1",
        tenant_id="t-1",
        role="viewer",
        secret_key=_JWT_SECRET,
    )


@pytest.fixture
def mock_session() -> AsyncMock:
    """Mock DB session."""
    session = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def app_with_mocks(mock_session: AsyncMock) -> MagicMock:
    """App with mocked DB."""
    app = create_app()

    async def _gen() -> AsyncGenerator[AsyncMock, None]:
        yield mock_session

    from app.infrastructure.database.connection import get_async_session

    app.dependency_overrides[get_async_session] = _gen
    return app


@pytest.fixture
async def voice_client(app_with_mocks: MagicMock) -> AsyncGenerator[AsyncClient, None]:
    """HTTP test client."""
    transport = ASGITransport(app=app_with_mocks)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


def _mock_tenant(config_json: dict | None = None) -> Tenant:
    """Create a mock tenant domain object."""
    return Tenant(
        id="t-1",
        name="Test Tenant",
        slug="test-tenant",
        config_json=config_json or {},
    )


# ── Voice Config Tests ───────────────────────────────────────────


class TestVoiceConfig:
    """Verify GET /api/v1/voice/config."""

    @pytest.mark.asyncio
    async def test_returns_voice_config_enabled(
        self,
        voice_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Returns voice availability for authenticated tenant."""
        with (
            patch("app.api.v1.voice.SQLTenantRepository") as mock_tenant_cls,
            patch("app.api.v1.voice.resolve_tenant_voice_config") as mock_resolve,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_tenant_repo = mock_tenant_cls.return_value
            mock_tenant_repo.get_by_id = AsyncMock(
                return_value=_mock_tenant({"voice_enabled": True})
            )
            mock_resolve.return_value = TenantVoiceConfig(
                voice_enabled=True,
                stt_provider="whisper",
                tts_provider="piper",
                tts_voice="en_US-lessac-medium",
                max_voice_sessions=3,
            )
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await voice_client.get(
                "/api/v1/voice/config",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["voice_enabled"] is True
        assert data["stt_provider"] == "whisper"
        assert data["tts_provider"] == "piper"

    @pytest.mark.asyncio
    async def test_returns_voice_config_disabled(
        self,
        voice_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Returns disabled when tenant has no voice config."""
        with (
            patch("app.api.v1.voice.SQLTenantRepository") as mock_tenant_cls,
            patch("app.api.v1.voice.resolve_tenant_voice_config") as mock_resolve,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_tenant_repo = mock_tenant_cls.return_value
            mock_tenant_repo.get_by_id = AsyncMock(return_value=_mock_tenant())
            mock_resolve.return_value = TenantVoiceConfig()
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await voice_client.get(
                "/api/v1/voice/config",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["voice_enabled"] is False

    @pytest.mark.asyncio
    async def test_tenant_not_found_returns_disabled(
        self,
        voice_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Returns disabled when tenant record doesn't exist."""
        with (
            patch("app.api.v1.voice.SQLTenantRepository") as mock_tenant_cls,
            patch("app.api.v1.voice.resolve_tenant_voice_config") as mock_resolve,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_tenant_repo = mock_tenant_cls.return_value
            mock_tenant_repo.get_by_id = AsyncMock(return_value=None)
            mock_resolve.return_value = TenantVoiceConfig()
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await voice_client.get(
                "/api/v1/voice/config",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["voice_enabled"] is False

    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(
        self,
        voice_client: AsyncClient,
    ) -> None:
        """No token returns 401."""
        response = await voice_client.get("/api/v1/voice/config")
        assert response.status_code == 401


# ── Voice Health Tests ───────────────────────────────────────────


class TestVoiceHealth:
    """Verify GET /api/v1/voice/health."""

    @pytest.mark.asyncio
    async def test_returns_health_status(
        self,
        voice_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Returns STT/TTS health status."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await voice_client.get(
                "/api/v1/voice/health",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "stt_available" in data
        assert "tts_available" in data

    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(
        self,
        voice_client: AsyncClient,
    ) -> None:
        """No token returns 401."""
        response = await voice_client.get("/api/v1/voice/health")
        assert response.status_code == 401
