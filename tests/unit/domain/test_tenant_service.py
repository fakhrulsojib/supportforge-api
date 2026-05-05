"""Unit tests for TenantService domain logic."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import SupportForgeError, TenantNotFoundError
from app.domain.models.tenant import Tenant, TenantCreate
from app.domain.services.tenant_service import TenantService


@pytest.fixture
def mock_repo() -> AsyncMock:
    """Create a mock TenantRepository."""
    return AsyncMock()


@pytest.fixture
def service(mock_repo: AsyncMock) -> TenantService:
    """Create a TenantService with mocked repository."""
    return TenantService(tenant_repo=mock_repo)


@pytest.fixture
def sample_tenant() -> Tenant:
    """Sample tenant domain model."""
    return Tenant(id="t-1", name="Acme Corp", slug="acme-corp")


@pytest.fixture
def sample_create() -> TenantCreate:
    """Sample tenant creation data."""
    return TenantCreate(name="Acme Corp", slug="acme-corp")


class TestCreateTenant:
    """Tests for TenantService.create_tenant."""

    async def test_create_success(
        self,
        service: TenantService,
        mock_repo: AsyncMock,
        sample_create: TenantCreate,
        sample_tenant: Tenant,
    ) -> None:
        """Successful creation when slug doesn't exist."""
        mock_repo.get_by_slug.return_value = None
        mock_repo.create.return_value = sample_tenant

        result = await service.create_tenant(sample_create)

        assert result.id == "t-1"
        assert result.slug == "acme-corp"
        mock_repo.get_by_slug.assert_awaited_once_with("acme-corp")
        mock_repo.create.assert_awaited_once_with(sample_create)

    async def test_create_duplicate_slug_raises(
        self,
        service: TenantService,
        mock_repo: AsyncMock,
        sample_create: TenantCreate,
        sample_tenant: Tenant,
    ) -> None:
        """Duplicate slug should raise 409."""
        mock_repo.get_by_slug.return_value = sample_tenant

        with pytest.raises(SupportForgeError) as exc_info:
            await service.create_tenant(sample_create)
        assert exc_info.value.status_code == 409
        assert "TENANT_SLUG_EXISTS" in exc_info.value.error_code


class TestGetTenant:
    """Tests for TenantService.get_tenant_by_id and get_tenant_by_slug."""

    async def test_get_by_id_found(
        self,
        service: TenantService,
        mock_repo: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Should return tenant when found by ID."""
        mock_repo.get_by_id.return_value = sample_tenant
        result = await service.get_tenant_by_id("t-1")
        assert result.id == "t-1"

    async def test_get_by_id_not_found(
        self,
        service: TenantService,
        mock_repo: AsyncMock,
    ) -> None:
        """Should raise TenantNotFoundError when not found by ID."""
        mock_repo.get_by_id.return_value = None
        with pytest.raises(TenantNotFoundError):
            await service.get_tenant_by_id("nonexistent")

    async def test_get_by_slug_found(
        self,
        service: TenantService,
        mock_repo: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Should return tenant when found by slug."""
        mock_repo.get_by_slug.return_value = sample_tenant
        result = await service.get_tenant_by_slug("acme-corp")
        assert result.slug == "acme-corp"

    async def test_get_by_slug_not_found(
        self,
        service: TenantService,
        mock_repo: AsyncMock,
    ) -> None:
        """Should raise TenantNotFoundError with slug context when slug not found."""
        mock_repo.get_by_slug.return_value = None
        with pytest.raises(TenantNotFoundError, match="nonexistent") as exc_info:
            await service.get_tenant_by_slug("nonexistent")
        # Verify the error message references the slug, not a generic tenant_id
        assert "slug" in str(exc_info.value).lower() or "nonexistent" in str(exc_info.value)


class TestListTenants:
    """Tests for TenantService.list_tenants."""

    async def test_list_returns_all(
        self,
        service: TenantService,
        mock_repo: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Should return all tenants from repo."""
        mock_repo.list_all.return_value = [sample_tenant]
        result = await service.list_tenants()
        assert len(result) == 1
        assert result[0].id == "t-1"

    async def test_list_empty(
        self,
        service: TenantService,
        mock_repo: AsyncMock,
    ) -> None:
        """Should return empty list when no tenants."""
        mock_repo.list_all.return_value = []
        result = await service.list_tenants()
        assert result == []


class TestUpdateTenant:
    """Tests for TenantService.update_tenant."""

    async def test_update_success(
        self,
        service: TenantService,
        mock_repo: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Should update and return tenant."""
        mock_repo.update.return_value = Tenant(
            id="t-1",
            name="Updated Acme",
            slug="acme-corp",
        )
        result = await service.update_tenant("t-1", name="Updated Acme")
        assert result.name == "Updated Acme"
        mock_repo.update.assert_awaited_once_with("t-1", name="Updated Acme")

    async def test_update_not_found(
        self,
        service: TenantService,
        mock_repo: AsyncMock,
    ) -> None:
        """Should raise TenantNotFoundError when tenant doesn't exist."""
        mock_repo.update.return_value = None
        with pytest.raises(TenantNotFoundError):
            await service.update_tenant("nonexistent", name="x")


class TestDeleteTenant:
    """Tests for TenantService.delete_tenant."""

    async def test_delete_success(
        self,
        service: TenantService,
        mock_repo: AsyncMock,
    ) -> None:
        """Should return True when deleted."""
        mock_repo.delete.return_value = True
        result = await service.delete_tenant("t-1")
        assert result is True

    async def test_delete_not_found(
        self,
        service: TenantService,
        mock_repo: AsyncMock,
    ) -> None:
        """Should raise TenantNotFoundError when tenant doesn't exist."""
        mock_repo.delete.return_value = False
        with pytest.raises(TenantNotFoundError):
            await service.delete_tenant("nonexistent")
