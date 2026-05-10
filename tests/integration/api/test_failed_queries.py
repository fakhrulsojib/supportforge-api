"""Integration tests for failed queries admin endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.security import create_access_token
from app.domain.models.enums import (
    EscalationTrigger,
    FailureReason,
    UserRole,
)
from app.domain.models.failed_query import FailedQuery
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
async def fq_client(app_with_mocks: MagicMock) -> AsyncGenerator[AsyncClient, None]:
    """HTTP test client."""
    transport = ASGITransport(app=app_with_mocks)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest.fixture
def sample_failed_query() -> FailedQuery:
    """A sample failed query."""
    return FailedQuery(
        id="fq-1",
        tenant_id="t-1",
        conversation_id="conv-1",
        message_id="msg-1",
        query_text="Where is my order?",
        failure_reason=FailureReason.NO_DOCS,
        retrieved_doc_count=0,
        max_relevance_score=0.0,
        escalation_trigger=EscalationTrigger.NO_CONTEXT,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def resolved_failed_query() -> FailedQuery:
    """A resolved failed query."""
    now = datetime.now(timezone.utc)
    return FailedQuery(
        id="fq-1",
        tenant_id="t-1",
        conversation_id="conv-1",
        message_id="msg-1",
        query_text="Where is my order?",
        failure_reason=FailureReason.NO_DOCS,
        retrieved_doc_count=0,
        max_relevance_score=0.0,
        escalation_trigger=EscalationTrigger.NO_CONTEXT,
        created_at=now,
        resolved_at=now,
        resolved_by="admin-1",
    )


# ── RBAC Tests ───────────────────────────────────────────────────


class TestFailedQueryRBAC:
    """Viewer/unauthenticated users cannot access failed query endpoints."""

    @pytest.mark.asyncio
    async def test_viewer_cannot_list_failed_queries(
        self,
        fq_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Viewer role should get 401."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await fq_client.get(
                "/api/v1/admin/failed-queries",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_viewer_cannot_resolve_failed_query(
        self,
        fq_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Viewer role should get 401 for resolve endpoint."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await fq_client.patch(
                "/api/v1/admin/failed-queries/fq-1/resolve",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_viewer_cannot_get_stats(
        self,
        fq_client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
    ) -> None:
        """Viewer role should get 401 for stats."""
        with patch("app.core.dependencies.SQLUserRepository") as mock_user_cls:
            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=viewer_user)

            response = await fq_client.get(
                "/api/v1/admin/failed-queries/stats",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(
        self,
        fq_client: AsyncClient,
    ) -> None:
        """No token should get 401."""
        response = await fq_client.get("/api/v1/admin/failed-queries")
        assert response.status_code == 401


# ── List Tests ───────────────────────────────────────────────────


class TestListFailedQueries:
    """Tests for GET /api/v1/admin/failed-queries."""

    @pytest.mark.asyncio
    async def test_list_happy_path(
        self,
        fq_client: AsyncClient,
        admin_token: str,
        admin_user: User,
        sample_failed_query: FailedQuery,
    ) -> None:
        """Admin should see failed queries for their tenant."""
        with (
            patch("app.api.v1.failed_queries.SQLFailedQueryRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.list_by_tenant = AsyncMock(
                return_value=([sample_failed_query], 1),
            )

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await fq_client.get(
                "/api/v1/admin/failed-queries",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["id"] == "fq-1"
        assert data["items"][0]["query_text"] == "Where is my order?"
        assert data["items"][0]["failure_reason"] == "no_docs"

    @pytest.mark.asyncio
    async def test_list_empty(
        self,
        fq_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Should return empty list when no failed queries exist."""
        with (
            patch("app.api.v1.failed_queries.SQLFailedQueryRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.list_by_tenant = AsyncMock(return_value=([], 0))

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await fq_client.get(
                "/api/v1/admin/failed-queries",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["items"] == []

    @pytest.mark.asyncio
    async def test_list_with_failure_reason_filter(
        self,
        fq_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Should pass failure_reason filter to repository."""
        with (
            patch("app.api.v1.failed_queries.SQLFailedQueryRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.list_by_tenant = AsyncMock(return_value=([], 0))

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await fq_client.get(
                "/api/v1/admin/failed-queries?failure_reason=no_docs",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        call_kwargs = mock_repo.list_by_tenant.call_args
        assert call_kwargs.kwargs.get("failure_reason") == FailureReason.NO_DOCS

    @pytest.mark.asyncio
    async def test_list_with_resolved_filter(
        self,
        fq_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Should pass resolved filter to repository."""
        with (
            patch("app.api.v1.failed_queries.SQLFailedQueryRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.list_by_tenant = AsyncMock(return_value=([], 0))

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await fq_client.get(
                "/api/v1/admin/failed-queries?resolved=false",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        call_kwargs = mock_repo.list_by_tenant.call_args
        assert call_kwargs.kwargs.get("resolved") is False

    @pytest.mark.asyncio
    async def test_list_invalid_failure_reason_ignored(
        self,
        fq_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Invalid failure_reason should be ignored (treated as None)."""
        with (
            patch("app.api.v1.failed_queries.SQLFailedQueryRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.list_by_tenant = AsyncMock(return_value=([], 0))

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await fq_client.get(
                "/api/v1/admin/failed-queries?failure_reason=invalid_reason",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        call_kwargs = mock_repo.list_by_tenant.call_args
        assert call_kwargs.kwargs.get("failure_reason") is None


# ── Resolve Tests ────────────────────────────────────────────────


class TestResolveFailedQuery:
    """Tests for PATCH /api/v1/admin/failed-queries/{id}/resolve."""

    @pytest.mark.asyncio
    async def test_resolve_happy_path(
        self,
        fq_client: AsyncClient,
        admin_token: str,
        admin_user: User,
        sample_failed_query: FailedQuery,
        resolved_failed_query: FailedQuery,
    ) -> None:
        """Should resolve a failed query."""
        with (
            patch("app.api.v1.failed_queries.SQLFailedQueryRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=sample_failed_query)
            mock_repo.mark_resolved = AsyncMock(return_value=resolved_failed_query)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await fq_client.patch(
                "/api/v1/admin/failed-queries/fq-1/resolve",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "fq-1"
        assert data["resolved_by"] == "admin-1"
        assert data["resolved_at"] is not None

    @pytest.mark.asyncio
    async def test_resolve_not_found(
        self,
        fq_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Should return 404 for non-existent failed query."""
        with (
            patch("app.api.v1.failed_queries.SQLFailedQueryRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=None)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await fq_client.patch(
                "/api/v1/admin/failed-queries/nonexistent/resolve",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_resolve_cross_tenant(
        self,
        fq_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Should return 404 for failed query from different tenant."""
        other_tenant_fq = FailedQuery(
            id="fq-other",
            tenant_id="other-tenant",
            conversation_id="conv-other",
            query_text="test",
            failure_reason=FailureReason.NO_DOCS,
            created_at=datetime.now(timezone.utc),
        )

        with (
            patch("app.api.v1.failed_queries.SQLFailedQueryRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=other_tenant_fq)

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await fq_client.patch(
                "/api/v1/admin/failed-queries/fq-other/resolve",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 404


# ── Stats Tests ──────────────────────────────────────────────────


class TestFailedQueryStats:
    """Tests for GET /api/v1/admin/failed-queries/stats."""

    @pytest.mark.asyncio
    async def test_stats_happy_path(
        self,
        fq_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Should return aggregated stats."""
        with (
            patch("app.api.v1.failed_queries.SQLFailedQueryRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.count_unresolved = AsyncMock(return_value=5)
            mock_repo.get_stats = AsyncMock(return_value={
                "reason_breakdown": {"no_docs": 3, "timeout": 2},
                "top_queries": [{"query_text": "Where is my order?", "count": 5}],
                "daily_trend": [{"date": "2026-01-01", "count": 3}],
            })

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await fq_client.get(
                "/api/v1/admin/failed-queries/stats",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_unresolved"] == 5
        assert data["reason_breakdown"]["no_docs"] == 3
        assert len(data["top_queries"]) == 1
        assert data["top_queries"][0]["query_text"] == "Where is my order?"
        assert len(data["daily_trend"]) == 1

    @pytest.mark.asyncio
    async def test_stats_empty(
        self,
        fq_client: AsyncClient,
        admin_token: str,
        admin_user: User,
    ) -> None:
        """Should return zero stats when no data."""
        with (
            patch("app.api.v1.failed_queries.SQLFailedQueryRepository") as mock_repo_cls,
            patch("app.core.dependencies.SQLUserRepository") as mock_user_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.count_unresolved = AsyncMock(return_value=0)
            mock_repo.get_stats = AsyncMock(return_value={
                "reason_breakdown": {},
                "top_queries": [],
                "daily_trend": [],
            })

            mock_user_repo = mock_user_cls.return_value
            mock_user_repo.get_by_id = AsyncMock(return_value=admin_user)

            response = await fq_client.get(
                "/api/v1/admin/failed-queries/stats",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_unresolved"] == 0
        assert data["reason_breakdown"] == {}
        assert data["top_queries"] == []
