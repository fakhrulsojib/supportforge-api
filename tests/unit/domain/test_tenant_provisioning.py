"""Unit tests for Phase 10 — Tenant Provisioning (domain models + service).

Tests cover:
- TenantStatus enum values
- Tenant domain model status field
- TenantService status transitions (valid + invalid)
"""

from __future__ import annotations

import pytest

from app.domain.models.enums import TenantStatus
from app.domain.models.tenant import Tenant, TenantCreate


class TestTenantStatusEnum:
    """Tests for the TenantStatus enum."""

    def test_has_pending(self) -> None:
        assert TenantStatus.PENDING == "pending"

    def test_has_active(self) -> None:
        assert TenantStatus.ACTIVE == "active"

    def test_has_suspended(self) -> None:
        assert TenantStatus.SUSPENDED == "suspended"

    def test_has_archived(self) -> None:
        assert TenantStatus.ARCHIVED == "archived"

    def test_enum_member_count(self) -> None:
        assert len(TenantStatus) == 4

    def test_from_string(self) -> None:
        assert TenantStatus("active") == TenantStatus.ACTIVE

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            TenantStatus("deleted")


class TestTenantDomainModelWithStatus:
    """Tests for the Tenant domain model status field."""

    def test_default_status_is_active(self) -> None:
        tenant = Tenant(name="Test", slug="test-co")
        assert tenant.status == TenantStatus.ACTIVE

    def test_explicit_pending_status(self) -> None:
        tenant = Tenant(name="Test", slug="test-co", status=TenantStatus.PENDING)
        assert tenant.status == TenantStatus.PENDING

    def test_explicit_suspended_status(self) -> None:
        tenant = Tenant(name="Test", slug="test-co", status=TenantStatus.SUSPENDED)
        assert tenant.status == TenantStatus.SUSPENDED

    def test_explicit_archived_status(self) -> None:
        tenant = Tenant(name="Test", slug="test-co", status=TenantStatus.ARCHIVED)
        assert tenant.status == TenantStatus.ARCHIVED

    def test_status_in_model_dump(self) -> None:
        tenant = Tenant(name="Test", slug="test-co", status=TenantStatus.SUSPENDED)
        data = tenant.model_dump()
        assert data["status"] == TenantStatus.SUSPENDED

    def test_status_from_string_in_constructor(self) -> None:
        tenant = Tenant(name="Test", slug="test-co", status="suspended")  # type: ignore[arg-type]
        assert tenant.status == TenantStatus.SUSPENDED


class TestTenantCreateWithStatus:
    """Tests for the TenantCreate DTO status field."""

    def test_default_status_is_active(self) -> None:
        data = TenantCreate(name="Test", slug="test-co")
        assert data.status == TenantStatus.ACTIVE

    def test_explicit_status(self) -> None:
        data = TenantCreate(name="Test", slug="test-co", status=TenantStatus.PENDING)
        assert data.status == TenantStatus.PENDING


# ── Tenant Service transition tests ─────────────────────────────

from unittest.mock import AsyncMock

from app.core.exceptions import SupportForgeError, TenantNotFoundError
from app.domain.services.tenant_service import VALID_TRANSITIONS, TenantService


class TestTenantServiceStatusTransitions:
    """Tests for TenantService.update_tenant_status() transition validation."""

    def _make_service(self, tenant: Tenant | None = None) -> tuple[TenantService, AsyncMock]:
        repo = AsyncMock()
        repo.get_by_id = AsyncMock(return_value=tenant)
        repo.update_status = AsyncMock(return_value=tenant)
        return TenantService(tenant_repo=repo), repo

    @pytest.mark.asyncio
    async def test_pending_to_active(self) -> None:
        tenant = Tenant(id="t-1", name="Test", slug="test-co", status=TenantStatus.PENDING)
        updated = Tenant(id="t-1", name="Test", slug="test-co", status=TenantStatus.ACTIVE)
        service, repo = self._make_service(tenant)
        repo.update_status = AsyncMock(return_value=updated)
        result = await service.update_tenant_status("t-1", TenantStatus.ACTIVE)
        assert result.status == TenantStatus.ACTIVE
        repo.update_status.assert_called_once_with("t-1", TenantStatus.ACTIVE)

    @pytest.mark.asyncio
    async def test_active_to_suspended(self) -> None:
        tenant = Tenant(id="t-1", name="Test", slug="test-co", status=TenantStatus.ACTIVE)
        updated = Tenant(id="t-1", name="Test", slug="test-co", status=TenantStatus.SUSPENDED)
        service, repo = self._make_service(tenant)
        repo.update_status = AsyncMock(return_value=updated)
        result = await service.update_tenant_status("t-1", TenantStatus.SUSPENDED)
        assert result.status == TenantStatus.SUSPENDED

    @pytest.mark.asyncio
    async def test_active_to_archived(self) -> None:
        tenant = Tenant(id="t-1", name="Test", slug="test-co", status=TenantStatus.ACTIVE)
        updated = Tenant(id="t-1", name="Test", slug="test-co", status=TenantStatus.ARCHIVED)
        service, repo = self._make_service(tenant)
        repo.update_status = AsyncMock(return_value=updated)
        result = await service.update_tenant_status("t-1", TenantStatus.ARCHIVED)
        assert result.status == TenantStatus.ARCHIVED

    @pytest.mark.asyncio
    async def test_suspended_to_active(self) -> None:
        tenant = Tenant(id="t-1", name="Test", slug="test-co", status=TenantStatus.SUSPENDED)
        updated = Tenant(id="t-1", name="Test", slug="test-co", status=TenantStatus.ACTIVE)
        service, repo = self._make_service(tenant)
        repo.update_status = AsyncMock(return_value=updated)
        result = await service.update_tenant_status("t-1", TenantStatus.ACTIVE)
        assert result.status == TenantStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_suspended_to_archived(self) -> None:
        tenant = Tenant(id="t-1", name="Test", slug="test-co", status=TenantStatus.SUSPENDED)
        updated = Tenant(id="t-1", name="Test", slug="test-co", status=TenantStatus.ARCHIVED)
        service, repo = self._make_service(tenant)
        repo.update_status = AsyncMock(return_value=updated)
        result = await service.update_tenant_status("t-1", TenantStatus.ARCHIVED)
        assert result.status == TenantStatus.ARCHIVED

    @pytest.mark.asyncio
    async def test_archived_is_terminal(self) -> None:
        """Archived tenants cannot transition to any other status."""
        tenant = Tenant(id="t-1", name="Test", slug="test-co", status=TenantStatus.ARCHIVED)
        service, _repo = self._make_service(tenant)
        with pytest.raises(SupportForgeError) as exc_info:
            await service.update_tenant_status("t-1", TenantStatus.ACTIVE)
        assert exc_info.value.status_code == 400
        assert exc_info.value.error_code == "INVALID_STATUS_TRANSITION"

    @pytest.mark.asyncio
    async def test_pending_to_suspended_invalid(self) -> None:
        """Pending cannot go directly to suspended."""
        tenant = Tenant(id="t-1", name="Test", slug="test-co", status=TenantStatus.PENDING)
        service, _repo = self._make_service(tenant)
        with pytest.raises(SupportForgeError) as exc_info:
            await service.update_tenant_status("t-1", TenantStatus.SUSPENDED)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_active_to_pending_invalid(self) -> None:
        """Active cannot go back to pending."""
        tenant = Tenant(id="t-1", name="Test", slug="test-co", status=TenantStatus.ACTIVE)
        service, _repo = self._make_service(tenant)
        with pytest.raises(SupportForgeError) as exc_info:
            await service.update_tenant_status("t-1", TenantStatus.PENDING)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_not_found_raises(self) -> None:
        service, _repo = self._make_service(None)
        with pytest.raises(TenantNotFoundError):
            await service.update_tenant_status("t-missing", TenantStatus.ACTIVE)

    def test_valid_transitions_map_completeness(self) -> None:
        """Every TenantStatus value must have an entry in VALID_TRANSITIONS."""
        for status in TenantStatus:
            assert status in VALID_TRANSITIONS

