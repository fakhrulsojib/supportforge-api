"""Tenant API schemas — request/response DTOs."""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003 — Pydantic needs runtime access

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from app.domain.models.enums import TenantStatus  # noqa: TCH001 — Pydantic runtime


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

    @model_validator(mode="after")
    def check_at_least_one_field(self) -> Self:
        """M5: Reject empty PATCH bodies — at least one field must be provided."""
        if self.name is None and self.config_json is None:
            msg = "At least one field (name or config_json) must be provided"
            raise ValueError(msg)
        return self


class TenantResponse(BaseModel):
    """Response body for tenant data."""

    id: str = Field(..., description="Tenant UUID")
    name: str = Field(..., description="Tenant display name")
    slug: str = Field(..., description="URL-safe tenant slug")
    status: TenantStatus | None = Field(None, description="Tenant lifecycle status")
    config_json: dict[str, object] | None = Field(None, description="Tenant configuration")
    created_at: datetime | None = Field(None, description="Creation timestamp")


class TenantListResponse(BaseModel):
    """Response body for tenant list."""

    tenants: list[TenantResponse] = Field(..., description="List of tenants")
    total: int = Field(..., description="Total count")


class TestHookRequest(BaseModel):
    """Request body for POST /api/v1/tenants/{id}/test-hook."""

    event_type: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Event type to test (e.g., on_escalation)",
    )
    url: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Webhook URL to test",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Custom headers to include in the test request",
    )


class TestHookResponse(BaseModel):
    """Response body for POST /api/v1/tenants/{id}/test-hook."""

    success: bool = Field(..., description="Whether the test request succeeded")
    status_code: int | None = Field(None, description="HTTP status code from the webhook")
    error: str | None = Field(None, description="Error message if the test failed")

