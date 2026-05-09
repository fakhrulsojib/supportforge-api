"""Tenant management API router — CRUD with RBAC.

Admin-only endpoints for creating, updating, and deleting tenants.
Slug-based retrieval is available to any authenticated user.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends

from app.api.schemas.tenant import (
    TenantCreateRequest,
    TenantListResponse,
    TenantResponse,
    TenantUpdateRequest,
)
from app.core.dependencies import get_current_user, require_role
from app.core.exceptions import TenantNotFoundError
from app.domain.models.enums import UserRole
from app.domain.models.tenant import TenantCreate
from app.domain.services.tenant_service import TenantService
from app.infrastructure.database.connection import get_async_session
from app.infrastructure.database.repositories.tenant_repo import SQLTenantRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.domain.models.user import User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/tenants", tags=["tenants"])


def _get_tenant_service(session: AsyncSession) -> TenantService:
    """Wire the TenantService with its repository dependency."""
    return TenantService(tenant_repo=SQLTenantRepository(session))


@router.post("/", response_model=TenantResponse, status_code=201)
async def create_tenant(
    request: TenantCreateRequest,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> TenantResponse:
    """Create a new tenant (admin only).

    Args:
        request: Tenant creation data.
        session: Database session.
        user: Authenticated admin user.

    Returns:
        Created TenantResponse.
    """
    service = _get_tenant_service(session)
    tenant_data = TenantCreate(
        name=request.name,
        slug=request.slug,
        config_json=request.config_json or {},
    )
    tenant = await service.create_tenant(tenant_data)
    logger.info("tenant_created_via_api", tenant_id=tenant.id, created_by=user.id)
    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        slug=tenant.slug,
        status=tenant.status,
        config_json=tenant.config_json,
        created_at=tenant.created_at,
    )


@router.get("/", response_model=TenantListResponse)
async def list_tenants(
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> TenantListResponse:
    """List all tenants (admin only).

    Args:
        session: Database session.
        user: Authenticated admin user.

    Returns:
        TenantListResponse with all tenants.
    """
    service = _get_tenant_service(session)
    tenants = await service.list_tenants()
    return TenantListResponse(
        tenants=[
            TenantResponse(
                id=t.id,
                name=t.name,
                slug=t.slug,
                status=t.status,
                config_json=t.config_json,
                created_at=t.created_at,
            )
            for t in tenants
        ],
        total=len(tenants),
    )


@router.get("/{slug_or_id}", response_model=TenantResponse)
async def get_tenant(
    slug_or_id: str,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_user),
) -> TenantResponse:
    """Get a tenant by slug or ID (any authenticated user).

    Tries slug lookup first, then falls back to ID lookup.

    Args:
        slug_or_id: Tenant slug or UUID.
        session: Database session.
        user: Authenticated user.

    Returns:
        TenantResponse.
    """
    service = _get_tenant_service(session)
    tenant = None
    try:
        tenant = await service.get_tenant_by_slug(slug_or_id)
    except TenantNotFoundError:
        repo = SQLTenantRepository(session)
        tenant = await repo.get_by_id(slug_or_id)
    if not tenant:
        raise TenantNotFoundError(tenant_id=slug_or_id)
    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        slug=tenant.slug,
        status=tenant.status,
        config_json=tenant.config_json,
        created_at=tenant.created_at,
    )


@router.patch("/{tenant_id}", response_model=TenantResponse)
async def update_tenant(
    tenant_id: str,
    request: TenantUpdateRequest,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> TenantResponse:
    """Update a tenant (admin only).

    Args:
        tenant_id: Tenant UUID.
        request: Fields to update.
        session: Database session.
        user: Authenticated admin user.

    Returns:
        Updated TenantResponse.
    """
    service = _get_tenant_service(session)
    update_data = request.model_dump(exclude_unset=True)
    tenant = await service.update_tenant(tenant_id, **update_data)
    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        slug=tenant.slug,
        status=tenant.status,
        config_json=tenant.config_json,
        created_at=tenant.created_at,
    )


@router.delete("/{tenant_id}", status_code=204)
async def delete_tenant(
    tenant_id: str,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> None:
    """Delete a tenant (admin only).

    Args:
        tenant_id: Tenant UUID.
        session: Database session.
        user: Authenticated admin user.
    """
    service = _get_tenant_service(session)
    await service.delete_tenant(tenant_id)
    logger.info("tenant_deleted_via_api", tenant_id=tenant_id, deleted_by=user.id)
