"""Tenant API schemas — request/response DTOs."""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003 — Pydantic needs runtime access

from pydantic import BaseModel, Field


class TenantCreateRequest(BaseModel):
    """Request body for POST /api/v1/tenants."""

    name: str = Field(..., min_length=1, max_length=255, description="Tenant display name")
    slug: str = Field(
        ...,
        min_length=2,
        max_length=63,
        pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$",
        description="URL-safe tenant slug (lowercase, hyphens only)",
    )
    config_json: dict[str, object] | None = Field(None, description="Optional tenant configuration")


class TenantUpdateRequest(BaseModel):
    """Request body for PATCH /api/v1/tenants/{id}."""

    name: str | None = Field(None, min_length=1, max_length=255, description="Updated display name")
    config_json: dict[str, object] | None = Field(None, description="Updated configuration")


class TenantResponse(BaseModel):
    """Response body for tenant data."""

    id: str = Field(..., description="Tenant UUID")
    name: str = Field(..., description="Tenant display name")
    slug: str = Field(..., description="URL-safe tenant slug")
    config_json: dict[str, object] | None = Field(None, description="Tenant configuration")
    created_at: datetime | None = Field(None, description="Creation timestamp")


class TenantListResponse(BaseModel):
    """Response body for tenant list."""

    tenants: list[TenantResponse] = Field(..., description="List of tenants")
    total: int = Field(..., description="Total count")
