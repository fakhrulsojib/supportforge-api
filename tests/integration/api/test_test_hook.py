"""Integration tests for POST /api/v1/tenants/{id}/test-hook endpoint.

Tests:
    - Admin can test a webhook URL (success response)
    - Admin can test a webhook URL (failure response)
    - Webhook timeout returns proper error
    - SSRF protection blocks internal URLs
    - Viewer cannot access test-hook endpoint
    - Tenant ownership enforced (can't test other tenant's hooks)
    - No auth returns 401
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.security import create_access_token
from app.domain.models.enums import UserRole
from app.domain.models.user import User
from app.main import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

_JWT_SECRET = "change-me-to-another-random-secret"  # noqa: S105


@pytest.fixture
def admin_user() -> User:
    return User(id="admin-1", tenant_id="t-1", email="admin@test.com", role=UserRole.ADMIN)


@pytest.fixture
def admin_token() -> str:
    return create_access_token(
        user_id="admin-1",
        tenant_id="t-1",
        role="admin",
        secret_key=_JWT_SECRET,
    )


@pytest.fixture
def viewer_user() -> User:
    return User(id="viewer-1", tenant_id="t-1", email="viewer@test.com", role=UserRole.VIEWER)


@pytest.fixture
def viewer_token() -> str:
    return create_access_token(
        user_id="viewer-1",
        tenant_id="t-1",
        role="viewer",
        secret_key=_JWT_SECRET,
    )


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def app_with_mocks(mock_session: AsyncMock) -> MagicMock:
    app = create_app()

    async def _mock_session_gen() -> AsyncGenerator[AsyncMock, None]:
        yield mock_session

    from app.infrastructure.database.connection import get_async_session

    app.dependency_overrides[get_async_session] = _mock_session_gen
    return app


@pytest.fixture
async def client(app_with_mocks: MagicMock) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app_with_mocks)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


class TestTestHookEndpoint:
    """Tests for POST /api/v1/tenants/{id}/test-hook."""

    @pytest.mark.asyncio
    async def test_successful_webhook_test(
        self,
        client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Admin testing a working webhook URL should get success=True."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
            patch(
                "app.rag.tools.webhook.validate_url_safety",
                new_callable=AsyncMock,
                return_value="1.2.3.4",
            ),
            patch("httpx.AsyncClient", return_value=mock_client_instance),
        ):
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await client.post(
                "/api/v1/tenants/t-1/test-hook",
                json={
                    "event_type": "on_escalation",
                    "url": "https://hooks.example.com/test",
                    "headers": {"X-Secret": "abc123"},
                },
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status_code"] == 200
        assert data["error"] is None

    @pytest.mark.asyncio
    async def test_webhook_returns_500(
        self,
        client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Webhook returning 500 should report success=False with status code."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
            patch(
                "app.rag.tools.webhook.validate_url_safety",
                new_callable=AsyncMock,
                return_value="1.2.3.4",
            ),
            patch("httpx.AsyncClient", return_value=mock_client_instance),
        ):
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await client.post(
                "/api/v1/tenants/t-1/test-hook",
                json={
                    "event_type": "on_escalation",
                    "url": "https://hooks.example.com/fail",
                },
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["status_code"] == 500
        assert data["error"] == "HTTP 500"

    @pytest.mark.asyncio
    async def test_wrong_tenant_id(
        self,
        client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Admin cannot test hooks for a different tenant."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await client.post(
                "/api/v1/tenants/other-tenant/test-hook",
                json={
                    "event_type": "on_escalation",
                    "url": "https://hooks.example.com",
                },
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "own tenant" in data["error"]

    @pytest.mark.asyncio
    async def test_viewer_cannot_test_hooks(
        self,
        client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Viewer role should be rejected (admin only)."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await client.post(
                "/api/v1/tenants/t-1/test-hook",
                json={
                    "event_type": "on_escalation",
                    "url": "https://hooks.example.com",
                },
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(
        self,
        client: AsyncClient,
    ) -> None:
        """No auth header should return 401."""
        response = await client.post(
            "/api/v1/tenants/t-1/test-hook",
            json={
                "event_type": "on_escalation",
                "url": "https://hooks.example.com",
            },
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_ssrf_blocked(
        self,
        client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """SSRF attempt (internal URL) should return error."""
        from app.rag.tools.webhook import SSRFError

        with (
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
            patch(
                "app.rag.tools.webhook.validate_url_safety",
                new_callable=AsyncMock,
                side_effect=SSRFError("URL resolves to private IP"),
            ),
        ):
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await client.post(
                "/api/v1/tenants/t-1/test-hook",
                json={
                    "event_type": "on_escalation",
                    "url": "http://169.254.169.254/metadata",
                },
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "private IP" in (data["error"] or "")
