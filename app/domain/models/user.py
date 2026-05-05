"""Domain model for users.

Pure Pydantic model — NO framework imports.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.models.enums import UserRole


class User(BaseModel):
    """A user belonging to a tenant."""

    id: str = ""
    tenant_id: str = ""
    email: str = Field(..., min_length=1, max_length=320)
    password_hash: str = ""
    role: UserRole = UserRole.VIEWER
    created_at: datetime | None = None


class UserCreate(BaseModel):
    """DTO for creating a new user.

    Note: Does NOT carry a plaintext password. The password hash
    is passed separately to the repository to minimise the surface
    area where plaintext credentials exist in memory.
    """

    email: str = Field(..., min_length=1, max_length=320)
    role: UserRole = UserRole.VIEWER
