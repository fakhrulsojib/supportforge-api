"""Domain model for tenants.

Pure Pydantic model — NO framework imports (FastAPI, SQLAlchemy, etc.).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.models.enums import TenantStatus


class Tenant(BaseModel):
    """A tenant in the multi-tenant system.

    Each tenant has isolated data: conversations, documents,
    ChromaDB collections, and user accounts.
    """

    id: str = ""
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")
    config_json: dict[str, object] = Field(default_factory=dict)
    status: TenantStatus = TenantStatus.ACTIVE
    created_at: datetime | None = None


class TenantCreate(BaseModel):
    """DTO for creating a new tenant."""

    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")
    config_json: dict[str, object] = Field(default_factory=dict)
    status: TenantStatus = TenantStatus.ACTIVE
