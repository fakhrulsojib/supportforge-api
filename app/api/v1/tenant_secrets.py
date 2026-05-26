"""Tenant secrets API — encrypted credential management.

CRUD endpoints for managing tenant secrets (tool auth credentials).
Secrets are stored encrypted via Fernet.  The GET endpoint returns
only key names, never decrypted values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.dependencies import require_role
from app.domain.models.enums import UserRole
from app.infrastructure.database.connection import get_async_session
from app.infrastructure.database.repositories.tenant_secret_repo import (
    SQLTenantSecretRepository,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.domain.models.user import User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/tenants/{tenant_id}/secrets", tags=["tenant-secrets"])


# ── Request / Response schemas ────────────────────────────────────


class CreateSecretRequest(BaseModel):
    """Create or update a tenant secret."""

    key: str = Field(..., min_length=1, max_length=255, examples=["tools_auth.default"])
    value: str = Field(..., min_length=1, examples=["Bearer sk-xxx"])


class SecretKeyListResponse(BaseModel):
    """List of secret key names (never values)."""

    keys: list[str]


class SecretActionResponse(BaseModel):
    """Confirmation response for secret operations."""

    key: str
    success: bool


# ── Helpers ───────────────────────────────────────────────────────


def _get_encryption_key() -> str:
    """Get the encryption key from settings."""
    from app.config import get_settings

    return get_settings().secret_key


def _verify_tenant_access(user: User, tenant_id: str) -> None:
    """Ensure the user has access to the specified tenant's secrets.

    Admin users can only manage secrets for their own tenant.
    Superadmins can manage any tenant.

    Raises:
        HTTPException: 403 if the user doesn't belong to the tenant.
    """
    if user.role == UserRole.SUPERADMIN:
        return  # Superadmins can access any tenant
    if user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only manage secrets for your own tenant",
        )


# ── Endpoints ─────────────────────────────────────────────────────


@router.post("", response_model=SecretActionResponse, status_code=201)
async def create_or_update_secret(
    tenant_id: str,
    request: CreateSecretRequest,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> SecretActionResponse:
    """Create or update an encrypted secret for a tenant.

    If the key already exists, the value is overwritten.

    Args:
        tenant_id: Tenant UUID.
        request: Secret key and plaintext value.
        session: Database session.
        user: Authenticated admin user.

    Returns:
        Confirmation with the key name.
    """
    _verify_tenant_access(user, tenant_id)
    repo = SQLTenantSecretRepository(
        session, encryption_key=_get_encryption_key()
    )
    await repo.upsert(tenant_id, key=request.key, value=request.value)
    await session.commit()
    logger.info(
        "tenant_secret_upserted",
        tenant_id=tenant_id,
        key=request.key,
        by=user.id,
    )
    return SecretActionResponse(key=request.key, success=True)


@router.get("", response_model=SecretKeyListResponse)
async def list_secret_keys(
    tenant_id: str,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> SecretKeyListResponse:
    """List secret key names for a tenant — never returns values.

    Args:
        tenant_id: Tenant UUID.
        session: Database session.
        user: Authenticated admin user.

    Returns:
        List of secret key names.
    """
    _verify_tenant_access(user, tenant_id)
    repo = SQLTenantSecretRepository(
        session, encryption_key=_get_encryption_key()
    )
    keys = await repo.list_keys(tenant_id)
    return SecretKeyListResponse(keys=keys)


@router.delete("/{key}")
async def delete_secret(
    tenant_id: str,
    key: str,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> SecretActionResponse:
    """Delete a tenant secret.

    Args:
        tenant_id: Tenant UUID.
        key: Secret key name to delete.
        session: Database session.
        user: Authenticated admin user.

    Returns:
        Confirmation. Raises 404 if key not found.
    """
    _verify_tenant_access(user, tenant_id)
    repo = SQLTenantSecretRepository(
        session, encryption_key=_get_encryption_key()
    )
    deleted = await repo.delete(tenant_id, key)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Secret '{key}' not found")
    await session.commit()
    logger.info(
        "tenant_secret_deleted",
        tenant_id=tenant_id,
        key=key,
        by=user.id,
    )
    return SecretActionResponse(key=key, success=True)
