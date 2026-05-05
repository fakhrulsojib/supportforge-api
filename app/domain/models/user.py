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
    """DTO for creating a new user."""

    email: str = Field(..., min_length=1, max_length=320)
    password: str = Field(..., min_length=8, max_length=128)
    role: UserRole = UserRole.VIEWER
