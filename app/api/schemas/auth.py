"""Authentication API schemas — request/response DTOs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    """Request body for POST /api/v1/auth/register."""

    email: str = Field(..., min_length=1, max_length=320, description="User email address")
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password (8-128 chars, mixed case, digit, special char)",
    )
    tenant_id: str = Field(..., min_length=1, description="Tenant the user is registering under")
    role: str = Field("viewer", description="User role: admin, agent, or viewer")


class LoginRequest(BaseModel):
    """Request body for POST /api/v1/auth/login."""

    email: str = Field(..., min_length=1, max_length=320, description="User email address")
    password: str = Field(..., min_length=1, description="User password")
    tenant_id: str = Field(..., min_length=1, description="Tenant context for login")


class RefreshRequest(BaseModel):
    """Request body for POST /api/v1/auth/refresh."""

    refresh_token: str = Field(..., min_length=1, description="Valid refresh token")


class TokenResponse(BaseModel):
    """Response body for successful authentication."""

    access_token: str = Field(..., description="JWT access token (15min TTL)")
    refresh_token: str = Field(..., description="JWT refresh token (7d TTL)")
    token_type: str = Field("bearer", description="Token type (always 'bearer')")
    expires_in: int = Field(..., description="Access token TTL in seconds")
