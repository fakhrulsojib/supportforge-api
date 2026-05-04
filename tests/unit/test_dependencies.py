"""Tests for dependency injection functions."""

from __future__ import annotations

import pytest
from starlette.datastructures import State

from app.core.dependencies import get_app_settings, get_tenant_id
from app.core.exceptions import TenantNotFoundError


class FakeRequest:
    """Minimal request-like object for testing."""

    def __init__(self, tenant_id: str | None = None) -> None:
        self.state = State()
        if tenant_id is not None:
            self.state.tenant_id = tenant_id


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
