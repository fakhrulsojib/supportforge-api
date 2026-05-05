"""Schema validation tests for tenant DTOs.

Tests edge cases for Pydantic field constraints:
  - slug pattern validation (lowercase, hyphens only)
  - min/max length enforcement
  - optional fields
  - config_json typing
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas.tenant import (
    TenantCreateRequest,
    TenantListResponse,
    TenantResponse,
    TenantUpdateRequest,
)


class TestTenantCreateRequest:
    """Edge-case validation for TenantCreateRequest."""

    def test_valid_request(self) -> None:
        req = TenantCreateRequest(name="Acme Corp", slug="acme-corp")
        assert req.config_json is None  # default

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            TenantCreateRequest(name="", slug="acme")

    def test_name_max_length(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            TenantCreateRequest(name="A" * 256, slug="acme")

    def test_slug_too_short(self) -> None:
        with pytest.raises(ValidationError, match="slug"):
            TenantCreateRequest(name="Acme", slug="a")

    def test_slug_max_length(self) -> None:
        with pytest.raises(ValidationError, match="slug"):
            TenantCreateRequest(name="Acme", slug="a" * 64)

    def test_slug_uppercase_rejected(self) -> None:
        with pytest.raises(ValidationError, match="slug"):
            TenantCreateRequest(name="Acme", slug="Acme")

    def test_slug_spaces_rejected(self) -> None:
        with pytest.raises(ValidationError, match="slug"):
            TenantCreateRequest(name="Acme", slug="acme corp")

    def test_slug_underscores_rejected(self) -> None:
        with pytest.raises(ValidationError, match="slug"):
            TenantCreateRequest(name="Acme", slug="acme_corp")

    def test_slug_valid_hyphenated(self) -> None:
        req = TenantCreateRequest(name="Acme", slug="acme-corp")
        assert req.slug == "acme-corp"

    def test_slug_valid_simple(self) -> None:
        req = TenantCreateRequest(name="Acme", slug="acme")
        assert req.slug == "acme"

    def test_slug_leading_hyphen_rejected(self) -> None:
        with pytest.raises(ValidationError, match="slug"):
            TenantCreateRequest(name="Acme", slug="-acme")

    def test_slug_trailing_hyphen_rejected(self) -> None:
        with pytest.raises(ValidationError, match="slug"):
            TenantCreateRequest(name="Acme", slug="acme-")

    def test_config_json_with_data(self) -> None:
        req = TenantCreateRequest(
            name="Acme",
            slug="acme",
            config_json={"theme": "dark", "max_docs": 100},
        )
        assert req.config_json == {"theme": "dark", "max_docs": 100}

    def test_missing_name_rejected(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            TenantCreateRequest(slug="acme")  # type: ignore[call-arg]

    def test_missing_slug_rejected(self) -> None:
        with pytest.raises(ValidationError, match="slug"):
            TenantCreateRequest(name="Acme")  # type: ignore[call-arg]


class TestTenantUpdateRequest:
    """Edge-case validation for TenantUpdateRequest."""

    def test_empty_update_rejected(self) -> None:
        """M5: PATCH with no fields should be rejected."""
        with pytest.raises(ValidationError, match="At least one field"):
            TenantUpdateRequest()

    def test_partial_update_name(self) -> None:
        req = TenantUpdateRequest(name="New Name")
        assert req.name == "New Name"
        assert req.config_json is None

    def test_partial_update_config(self) -> None:
        req = TenantUpdateRequest(config_json={"key": "val"})
        assert req.name is None
        assert req.config_json == {"key": "val"}

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            TenantUpdateRequest(name="")


class TestTenantResponse:
    """Edge-case validation for TenantResponse."""

    def test_valid_response(self) -> None:
        resp = TenantResponse(id="t-1", name="Acme", slug="acme")
        assert resp.config_json is None
        assert resp.created_at is None

    def test_missing_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="id"):
            TenantResponse(name="Acme", slug="acme")  # type: ignore[call-arg]


class TestTenantListResponse:
    """Edge-case validation for TenantListResponse."""

    def test_valid_list(self) -> None:
        resp = TenantListResponse(
            tenants=[TenantResponse(id="t-1", name="Acme", slug="acme")],
            total=1,
        )
        assert len(resp.tenants) == 1

    def test_empty_list(self) -> None:
        resp = TenantListResponse(tenants=[], total=0)
        assert resp.total == 0
