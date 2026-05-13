"""Integration tests for analytics API endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.security import create_access_token
from app.domain.models.analytics import DailyStatEntry, IntentEntry, SatisfactionSummary
from app.domain.models.enums import UserRole
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
    """Authenticated viewer user (non-admin)."""
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
async def analytics_client(app_with_mocks: MagicMock) -> AsyncGenerator[AsyncClient, None]:
    """HTTP test client."""
    transport = ASGITransport(app=app_with_mocks)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


# ── RBAC Tests ───────────────────────────────────────────────────


class TestAnalyticsRBAC:
    """Viewer/unauthenticated users cannot access analytics endpoints."""

    @pytest.mark.asyncio
    async def test_viewer_cannot_get_daily_stats(
        self,
        analytics_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Viewer role should get 401."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await analytics_client.get(
                "/api/v1/analytics/daily-stats",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_viewer_cannot_get_top_intents(
        self,
        analytics_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Viewer role should get 401 for top intents."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await analytics_client.get(
                "/api/v1/analytics/top-intents",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_viewer_cannot_get_satisfaction(
        self,
        analytics_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Viewer role should get 401 for satisfaction."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await analytics_client.get(
                "/api/v1/analytics/satisfaction",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(
        self,
        analytics_client: AsyncClient,
    ) -> None:
        """No token should get 401."""
        response = await analytics_client.get("/api/v1/analytics/daily-stats")
        assert response.status_code == 401


# ── Daily Stats Tests ────────────────────────────────────────────


class TestDailyStats:
    """Tests for GET /api/v1/analytics/daily-stats."""

    @pytest.mark.asyncio
    async def test_daily_stats_happy_path(
        self,
        analytics_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Admin should get daily stats with data."""
        mock_entries = [
            DailyStatEntry(date="2026-05-12", total_conversations=5, total_messages=10),
            DailyStatEntry(date="2026-05-13", total_conversations=3, total_messages=7),
        ]

        with (
            patch("app.api.v1.analytics.SQLAnalyticsRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_daily_stats = AsyncMock(return_value=mock_entries)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await analytics_client.get(
                "/api/v1/analytics/daily-stats?days=7",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data["stats"]) == 2
        assert data["stats"][0]["date"] == "2026-05-12"
        assert data["stats"][0]["total_conversations"] == 5
        assert data["stats"][0]["total_messages"] == 10
        assert data["stats"][1]["date"] == "2026-05-13"

    @pytest.mark.asyncio
    async def test_daily_stats_empty(
        self,
        analytics_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Empty stats returns empty array."""
        with (
            patch("app.api.v1.analytics.SQLAnalyticsRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_daily_stats = AsyncMock(return_value=[])

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await analytics_client.get(
                "/api/v1/analytics/daily-stats",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["stats"] == []

    @pytest.mark.asyncio
    async def test_daily_stats_days_param_passed(
        self,
        analytics_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Days parameter should be passed to service."""
        with (
            patch("app.api.v1.analytics.SQLAnalyticsRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_daily_stats = AsyncMock(return_value=[])

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await analytics_client.get(
                "/api/v1/analytics/daily-stats?days=90",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        # Verify days parameter was passed (clamped to max 365)
        mock_repo.get_daily_stats.assert_called_once()
        call_kwargs = mock_repo.get_daily_stats.call_args
        assert call_kwargs.kwargs.get("days") == 90 or call_kwargs[1].get("days") == 90


# ── Top Intents Tests ────────────────────────────────────────────


class TestTopIntents:
    """Tests for GET /api/v1/analytics/top-intents."""

    @pytest.mark.asyncio
    async def test_top_intents_happy_path(
        self,
        analytics_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Admin should get top intents with data."""
        mock_intents = [
            IntentEntry(name="shipping_policy.pdf", count=42),
            IntentEntry(name="returns_guide.md", count=15),
        ]

        with (
            patch("app.api.v1.analytics.SQLAnalyticsRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_top_intents = AsyncMock(return_value=mock_intents)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await analytics_client.get(
                "/api/v1/analytics/top-intents?limit=5",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data["intents"]) == 2
        assert data["intents"][0]["name"] == "shipping_policy.pdf"
        assert data["intents"][0]["count"] == 42

    @pytest.mark.asyncio
    async def test_top_intents_empty(
        self,
        analytics_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Empty intents returns empty array."""
        with (
            patch("app.api.v1.analytics.SQLAnalyticsRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_top_intents = AsyncMock(return_value=[])

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await analytics_client.get(
                "/api/v1/analytics/top-intents",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["intents"] == []


# ── Satisfaction Tests ───────────────────────────────────────────


class TestSatisfaction:
    """Tests for GET /api/v1/analytics/satisfaction."""

    @pytest.mark.asyncio
    async def test_satisfaction_happy_path(
        self,
        analytics_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Admin should get satisfaction metrics."""
        mock_summary = SatisfactionSummary(
            positive=80, negative=20, total=100, rate=0.8,
        )

        with (
            patch("app.api.v1.analytics.SQLAnalyticsRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_satisfaction_summary = AsyncMock(return_value=mock_summary)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await analytics_client.get(
                "/api/v1/analytics/satisfaction",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["positive"] == 80
        assert data["negative"] == 20
        assert data["total"] == 100
        assert data["rate"] == 0.8

    @pytest.mark.asyncio
    async def test_satisfaction_no_feedback(
        self,
        analytics_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """No feedback returns zero counts and 0.0 rate."""
        mock_summary = SatisfactionSummary(
            positive=0, negative=0, total=0, rate=0.0,
        )

        with (
            patch("app.api.v1.analytics.SQLAnalyticsRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_satisfaction_summary = AsyncMock(return_value=mock_summary)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await analytics_client.get(
                "/api/v1/analytics/satisfaction",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["rate"] == 0.0

    @pytest.mark.asyncio
    async def test_satisfaction_all_positive(
        self,
        analytics_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """100% positive feedback returns rate 1.0."""
        mock_summary = SatisfactionSummary(
            positive=50, negative=0, total=50, rate=1.0,
        )

        with (
            patch("app.api.v1.analytics.SQLAnalyticsRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_satisfaction_summary = AsyncMock(return_value=mock_summary)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await analytics_client.get(
                "/api/v1/analytics/satisfaction",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["rate"] == 1.0
        assert data["negative"] == 0
