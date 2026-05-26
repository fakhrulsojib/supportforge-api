"""Unit tests for widget API endpoints (Phase 4).

Tests cover:
- POST /api/v1/widget/session — session creation, embed-key validation,
  origin checking, rate limiting, and input validation.
- GET  /api/v1/widget/ui-config/{slug} — public UI config retrieval,
  defaults, and security (never exposes secrets/tools).
- _matches_domain helper — exact match, wildcards, port/protocol
  stripping, and empty-input edge cases.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.widget import _matches_domain, _rate_limit_store
from app.domain.models.enums import TenantStatus
from app.domain.models.tenant import Tenant
from app.main import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# ── Helpers & fixtures ───────────────────────────────────────────


def _make_tenant(
    *,
    tenant_id: str = "t-100",
    name: str = "MedForge",
    slug: str = "medforge",
    status: TenantStatus = TenantStatus.ACTIVE,
    config_json: dict | None = None,
) -> Tenant:
    """Build a Tenant domain object with sensible defaults."""
    return Tenant(
        id=tenant_id,
        name=name,
        slug=slug,
        status=status,
        config_json=config_json
        or {
            "embed_key": "pk_live_abc123",
            "embed_domains": [],
        },
    )


@asynccontextmanager
async def _mock_session_ctx(mock_repo: AsyncMock):
    """Async context manager that replaces ``AsyncSessionLocal()``.

    Yields a lightweight mock session whose ``SQLTenantRepository``
    construction returns ``mock_repo``.
    """
    yield MagicMock()


@pytest.fixture(autouse=True)
def _clear_rate_limits():
    """Reset the in-memory rate-limit store between tests."""
    _rate_limit_store.clear()
    yield
    _rate_limit_store.clear()


@pytest.fixture
def active_tenant() -> Tenant:
    """Active tenant with a valid embed key."""
    return _make_tenant()


@pytest.fixture
def inactive_tenant() -> Tenant:
    """Suspended tenant."""
    return _make_tenant(status=TenantStatus.SUSPENDED)


@pytest.fixture
def tenant_with_domains() -> Tenant:
    """Active tenant with embed_domains configured."""
    return _make_tenant(
        config_json={
            "embed_key": "pk_live_abc123",
            "embed_domains": ["*.medforge.com", "example.org"],
        },
    )


@pytest.fixture
def tenant_with_ui_config() -> Tenant:
    """Tenant with full ui_config block."""
    return _make_tenant(
        config_json={
            "embed_key": "pk_live_abc123",
            "agent_prompt": "SECRET PROMPT — must never leak",
            "tools": ["search", "escalate"],
            "secrets": {"api_key": "sk-secret"},
            "ui_config": {
                "brand_name": "MedForge Health",
                "logo_url": "https://cdn.medforge.com/logo.png",
                "welcome_message": "Welcome to MedForge!",
                "placeholder_text": "Ask a health question...",
                "theme": {"primary_color": "#ff0000"},
                "widget": {"position": "bottom-left"},
            },
        },
    )


@pytest.fixture
def tenant_no_embed_key() -> Tenant:
    """Active tenant without an embed key in config."""
    return _make_tenant(config_json={"embed_domains": []})


async def _build_client(mock_repo_instance: AsyncMock) -> AsyncGenerator[AsyncClient, None]:
    """Create an httpx AsyncClient with AsyncSessionLocal + repo mocked."""
    app = create_app()

    @asynccontextmanager
    async def _fake_session():
        yield MagicMock()

    with (
        patch("app.api.v1.widget.AsyncSessionLocal", _fake_session),
        patch("app.api.v1.widget.SQLTenantRepository", return_value=mock_repo_instance),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            yield ac


# ── POST /api/v1/widget/session ──────────────────────────────────


class TestCreateWidgetSession:
    """Tests for POST /api/v1/widget/session."""

    @pytest.mark.asyncio
    async def test_valid_slug_and_embed_key_returns_200(self, active_tenant: Tenant) -> None:
        """Happy path: valid slug + correct embed_key → 200 with ws_ token."""
        mock_repo = AsyncMock()
        mock_repo.get_by_slug = AsyncMock(return_value=active_tenant)

        async for client in _build_client(mock_repo):
            response = await client.post(
                "/api/v1/widget/session",
                json={"tenant_slug": "medforge", "embed_key": "pk_live_abc123"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["session_token"].startswith("ws_")
        assert data["tenant_id"] == "t-100"
        assert data["tenant_slug"] == "medforge"
        assert data["expires_in"] == 3600  # 60 min × 60 sec

    @pytest.mark.asyncio
    async def test_wrong_embed_key_returns_403(self, active_tenant: Tenant) -> None:
        """Wrong embed_key should be rejected with 403."""
        mock_repo = AsyncMock()
        mock_repo.get_by_slug = AsyncMock(return_value=active_tenant)

        async for client in _build_client(mock_repo):
            response = await client.post(
                "/api/v1/widget/session",
                json={"tenant_slug": "medforge", "embed_key": "wrong-key"},
            )

        assert response.status_code == 403
        assert "Invalid tenant or embed key" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_missing_embed_key_in_tenant_config_returns_403(
        self, tenant_no_embed_key: Tenant
    ) -> None:
        """Tenant with no embed_key configured should reject with 403."""
        mock_repo = AsyncMock()
        mock_repo.get_by_slug = AsyncMock(return_value=tenant_no_embed_key)

        async for client in _build_client(mock_repo):
            response = await client.post(
                "/api/v1/widget/session",
                json={"tenant_slug": "medforge", "embed_key": "pk_live_abc123"},
            )

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_tenant_not_found_returns_403(self) -> None:
        """Unknown slug → 403 (not 404 — avoid leaking tenant existence)."""
        mock_repo = AsyncMock()
        mock_repo.get_by_slug = AsyncMock(return_value=None)

        async for client in _build_client(mock_repo):
            response = await client.post(
                "/api/v1/widget/session",
                json={"tenant_slug": "nonexistent", "embed_key": "pk_live_abc123"},
            )

        assert response.status_code == 403
        assert "Invalid tenant or embed key" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_inactive_tenant_returns_403(self, inactive_tenant: Tenant) -> None:
        """Suspended tenant → 403."""
        mock_repo = AsyncMock()
        mock_repo.get_by_slug = AsyncMock(return_value=inactive_tenant)

        async for client in _build_client(mock_repo):
            response = await client.post(
                "/api/v1/widget/session",
                json={"tenant_slug": "medforge", "embed_key": "pk_live_abc123"},
            )

        assert response.status_code == 403
        assert "Tenant not active" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_missing_tenant_slug_returns_422(self) -> None:
        """Omitting tenant_slug should fail Pydantic validation → 422."""
        mock_repo = AsyncMock()
        mock_repo.get_by_slug = AsyncMock(return_value=None)

        async for client in _build_client(mock_repo):
            response = await client.post(
                "/api/v1/widget/session",
                json={"embed_key": "pk_live_abc123"},
            )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_embed_key_returns_422(self) -> None:
        """Empty embed_key string should fail min_length=1 → 422."""
        mock_repo = AsyncMock()
        mock_repo.get_by_slug = AsyncMock(return_value=None)

        async for client in _build_client(mock_repo):
            response = await client.post(
                "/api/v1/widget/session",
                json={"tenant_slug": "medforge", "embed_key": ""},
            )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_origin_matches_embed_domains_returns_200(
        self, tenant_with_domains: Tenant
    ) -> None:
        """Origin matching embed_domains should be accepted."""
        mock_repo = AsyncMock()
        mock_repo.get_by_slug = AsyncMock(return_value=tenant_with_domains)

        async for client in _build_client(mock_repo):
            response = await client.post(
                "/api/v1/widget/session",
                json={"tenant_slug": "medforge", "embed_key": "pk_live_abc123"},
                headers={"Origin": "https://app.medforge.com"},
            )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_origin_does_not_match_embed_domains_returns_403(
        self, tenant_with_domains: Tenant
    ) -> None:
        """Origin NOT in embed_domains → 403."""
        mock_repo = AsyncMock()
        mock_repo.get_by_slug = AsyncMock(return_value=tenant_with_domains)

        async for client in _build_client(mock_repo):
            response = await client.post(
                "/api/v1/widget/session",
                json={"tenant_slug": "medforge", "embed_key": "pk_live_abc123"},
                headers={"Origin": "https://evil.com"},
            )

        assert response.status_code == 403
        assert "Origin not allowed" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_no_embed_domains_allows_any_origin(self, active_tenant: Tenant) -> None:
        """Empty embed_domains list → any origin allowed."""
        mock_repo = AsyncMock()
        mock_repo.get_by_slug = AsyncMock(return_value=active_tenant)

        async for client in _build_client(mock_repo):
            response = await client.post(
                "/api/v1/widget/session",
                json={"tenant_slug": "medforge", "embed_key": "pk_live_abc123"},
                headers={"Origin": "https://anything.example.com"},
            )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_rate_limiting_returns_429(self, active_tenant: Tenant) -> None:
        """Exceeding the per-IP rate limit → 429."""
        mock_repo = AsyncMock()
        mock_repo.get_by_slug = AsyncMock(return_value=active_tenant)

        async for client in _build_client(mock_repo):
            # Default limit is 10 per minute. Fire 10 allowed requests.
            for _ in range(10):
                resp = await client.post(
                    "/api/v1/widget/session",
                    json={"tenant_slug": "medforge", "embed_key": "pk_live_abc123"},
                )
                assert resp.status_code == 200

            # 11th request should be rate-limited
            response = await client.post(
                "/api/v1/widget/session",
                json={"tenant_slug": "medforge", "embed_key": "pk_live_abc123"},
            )

        assert response.status_code == 429
        assert "Too many session requests" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_visitor_id_included_in_token(self, active_tenant: Tenant) -> None:
        """When visitor_id is provided, it should be part of the JWT payload."""
        mock_repo = AsyncMock()
        mock_repo.get_by_slug = AsyncMock(return_value=active_tenant)

        async for client in _build_client(mock_repo):
            response = await client.post(
                "/api/v1/widget/session",
                json={
                    "tenant_slug": "medforge",
                    "embed_key": "pk_live_abc123",
                    "visitor_id": "visitor-xyz-999",
                },
            )

        assert response.status_code == 200
        token = response.json()["session_token"]
        assert token.startswith("ws_")

        # Decode and verify visitor_id is embedded
        from app.config import get_settings
        from app.core.widget_token import verify_widget_token

        settings = get_settings()
        payload = verify_widget_token(token, settings.jwt_secret_key)
        assert payload.visitor_id == "visitor-xyz-999"


# ── GET /api/v1/widget/ui-config/{slug} ─────────────────────────


class TestGetWidgetUIConfig:
    """Tests for GET /api/v1/widget/ui-config/{slug}."""

    @pytest.mark.asyncio
    async def test_valid_slug_returns_200_with_ui_config(
        self, tenant_with_ui_config: Tenant
    ) -> None:
        """Active tenant with ui_config → 200 with correct fields."""
        mock_repo = AsyncMock()
        mock_repo.get_by_slug = AsyncMock(return_value=tenant_with_ui_config)

        async for client in _build_client(mock_repo):
            response = await client.get("/api/v1/widget/ui-config/medforge")

        assert response.status_code == 200
        data = response.json()
        assert data["brand_name"] == "MedForge Health"
        assert data["logo_url"] == "https://cdn.medforge.com/logo.png"
        assert data["welcome_message"] == "Welcome to MedForge!"
        assert data["placeholder_text"] == "Ask a health question..."
        assert data["theme"]["primary_color"] == "#ff0000"
        assert data["widget"]["position"] == "bottom-left"

    @pytest.mark.asyncio
    async def test_unknown_slug_returns_404(self) -> None:
        """Unknown slug → 404."""
        mock_repo = AsyncMock()
        mock_repo.get_by_slug = AsyncMock(return_value=None)

        async for client in _build_client(mock_repo):
            response = await client.get("/api/v1/widget/ui-config/unknown-slug")

        assert response.status_code == 404
        assert "Tenant not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_inactive_tenant_returns_404(self, inactive_tenant: Tenant) -> None:
        """Inactive tenant → 404 (not 403, to hide existence)."""
        mock_repo = AsyncMock()
        mock_repo.get_by_slug = AsyncMock(return_value=inactive_tenant)

        async for client in _build_client(mock_repo):
            response = await client.get("/api/v1/widget/ui-config/medforge")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_default_values_when_no_ui_config(self, active_tenant: Tenant) -> None:
        """Tenant with no ui_config block → 200 with sensible defaults."""
        mock_repo = AsyncMock()
        mock_repo.get_by_slug = AsyncMock(return_value=active_tenant)

        async for client in _build_client(mock_repo):
            response = await client.get("/api/v1/widget/ui-config/medforge")

        assert response.status_code == 200
        data = response.json()
        assert data["brand_name"] == "Support"
        assert data["welcome_message"] == "Hi! How can I help you today?"
        assert data["placeholder_text"] == "Type your message..."
        assert data["theme"] == {}
        assert data["widget"] == {}

    @pytest.mark.asyncio
    async def test_never_exposes_secrets_or_agent_prompt(
        self, tenant_with_ui_config: Tenant
    ) -> None:
        """ui-config endpoint must NEVER leak tools, secrets, or agent_prompt."""
        mock_repo = AsyncMock()
        mock_repo.get_by_slug = AsyncMock(return_value=tenant_with_ui_config)

        async for client in _build_client(mock_repo):
            response = await client.get("/api/v1/widget/ui-config/medforge")

        assert response.status_code == 200
        data = response.json()
        # These internal keys must not appear anywhere in the response
        raw = response.text
        assert "agent_prompt" not in raw
        assert "SECRET PROMPT" not in raw
        assert "tools" not in data
        assert "secrets" not in data
        assert "sk-secret" not in raw


# ── _matches_domain helper ───────────────────────────────────────


class TestMatchesDomain:
    """Tests for the _matches_domain() helper function."""

    def test_exact_match(self) -> None:
        """Exact domain string should match."""
        assert _matches_domain("https://example.com", ["example.com"]) is True

    def test_wildcard_subdomain_match(self) -> None:
        """*.medforge.com should match sub.medforge.com."""
        assert _matches_domain("https://sub.medforge.com", ["*.medforge.com"]) is True

    def test_wildcard_matches_bare_domain(self) -> None:
        """*.medforge.com should also match bare medforge.com."""
        assert _matches_domain("https://medforge.com", ["*.medforge.com"]) is True

    def test_no_match(self) -> None:
        """Domain not in allowed list should not match."""
        assert _matches_domain("https://evil.com", ["example.com", "*.medforge.com"]) is False

    def test_empty_origin_rejected_when_domains_configured(self) -> None:
        """Empty origin → rejected when domain restriction is active."""
        assert _matches_domain("", ["example.com"]) is False

    def test_empty_allowed_domains_allows(self) -> None:
        """Empty allowed_domains list → no restriction (True)."""
        assert _matches_domain("https://anything.com", []) is True

    def test_both_empty_allows(self) -> None:
        """Both empty → True."""
        assert _matches_domain("", []) is True

    def test_port_stripping(self) -> None:
        """Port in origin URL should be stripped before comparison."""
        assert _matches_domain("https://example.com:8443", ["example.com"]) is True

    def test_protocol_stripping_http(self) -> None:
        """http:// prefix should be stripped."""
        assert _matches_domain("http://example.com", ["example.com"]) is True

    def test_protocol_stripping_https(self) -> None:
        """https:// prefix should be stripped."""
        assert _matches_domain("https://example.com", ["example.com"]) is True

    def test_case_insensitive_match(self) -> None:
        """Domain comparison should be case-insensitive."""
        assert _matches_domain("https://Example.COM", ["example.com"]) is True

    def test_wildcard_deep_subdomain(self) -> None:
        """*.medforge.com should match deeply nested subdomains."""
        assert _matches_domain("https://a.b.c.medforge.com", ["*.medforge.com"]) is True

    def test_wildcard_does_not_match_different_tld(self) -> None:
        """*.medforge.com should NOT match medforge.org."""
        assert _matches_domain("https://medforge.org", ["*.medforge.com"]) is False

    def test_multiple_allowed_domains(self) -> None:
        """Should match if origin matches ANY allowed domain."""
        allowed = ["example.com", "*.medforge.com", "other.org"]
        assert _matches_domain("https://other.org", allowed) is True
        assert _matches_domain("https://app.medforge.com", allowed) is True
        assert _matches_domain("https://nope.net", allowed) is False
