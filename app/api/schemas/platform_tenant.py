"""Platform tenant API schemas — request/response DTOs for superadmin tenant provisioning."""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003 — Pydantic needs runtime access

from pydantic import BaseModel, Field

from app.domain.models.enums import TenantStatus  # noqa: TCH001 — Pydantic runtime


class PlatformTenantCreateRequest(BaseModel):
    """Request body for POST /api/v1/platform/tenants."""

    name: str = Field(..., min_length=1, max_length=255, description="Tenant display name")
    slug: str = Field(
        ...,
        min_length=2,
        max_length=63,
        pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$",
        description="URL-safe tenant slug (lowercase, hyphens only)",
    )
    config_json: dict[str, object] | None = Field(None, description="Optional tenant configuration")


class PlatformTenantResponse(BaseModel):
    """Response body for platform tenant data (includes status)."""

    id: str = Field(..., description="Tenant UUID")
    name: str = Field(..., description="Tenant display name")
    slug: str = Field(..., description="URL-safe tenant slug")
    status: TenantStatus = Field(..., description="Tenant lifecycle status")
    config_json: dict[str, object] | None = Field(None, description="Tenant configuration")
    created_at: datetime | None = Field(None, description="Creation timestamp")


class PlatformTenantListResponse(BaseModel):
    """Response body for paginated platform tenant list."""

    tenants: list[PlatformTenantResponse] = Field(..., description="List of tenants")
    total: int = Field(..., description="Total count matching filter")


class TenantStatusUpdateRequest(BaseModel):
    """Request body for PATCH /api/v1/platform/tenants/{id}/status."""

    status: TenantStatus = Field(..., description="New tenant status")
