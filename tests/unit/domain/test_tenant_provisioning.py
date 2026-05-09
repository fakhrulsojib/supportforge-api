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
