"""Platform tenant management API router — superadmin-only CRUD.

Provides tenant provisioning endpoints under ``/api/v1/platform/tenants``.
All endpoints require superadmin authentication via ``require_superadmin()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends, Query

from app.api.schemas.platform_tenant import (
    PlatformTenantCreateRequest,
    PlatformTenantListResponse,
    PlatformTenantResponse,
    TenantStatusUpdateRequest,
)
from app.core.dependencies import require_superadmin
from app.domain.models.enums import TenantStatus
from app.domain.models.tenant import TenantCreate
from app.domain.services.tenant_service import TenantService
from app.infrastructure.database.connection import get_async_session
from app.infrastructure.database.repositories.tenant_repo import SQLTenantRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.domain.models.user import User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/platform/tenants", tags=["platform-tenants"])


def _get_tenant_service(session: AsyncSession) -> TenantService:
    """Wire the TenantService with its repository dependency."""
    return TenantService(tenant_repo=SQLTenantRepository(session))


@router.post("/", response_model=PlatformTenantResponse, status_code=201)
async def platform_create_tenant(
    request: PlatformTenantCreateRequest,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_superadmin()),
) -> PlatformTenantResponse:
    """Create a new tenant (superadmin only).

    New tenants are created with ``status=active`` by default.

    Args:
        request: Tenant creation data.
        session: Database session.
        user: Authenticated superadmin user.

    Returns:
        Created PlatformTenantResponse.
    """
    service = _get_tenant_service(session)
    tenant_data = TenantCreate(
        name=request.name,
        slug=request.slug,
        config_json=request.config_json or {},
    )
    tenant = await service.create_tenant(tenant_data)
    logger.info(
        "platform_tenant_created",
        tenant_id=tenant.id,
        created_by=user.id,
    )
    return PlatformTenantResponse(
        id=tenant.id,
        name=tenant.name,
        slug=tenant.slug,
        status=tenant.status,
        config_json=tenant.config_json,
        created_at=tenant.created_at,
    )


@router.get("/", response_model=PlatformTenantListResponse)
async def platform_list_tenants(
    status: TenantStatus | None = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Page size"),
    offset: int = Query(0, ge=0, description="Page offset"),
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_superadmin()),
) -> PlatformTenantListResponse:
    """List tenants with optional status filter (superadmin only).

    Args:
        status: Optional status filter.
        limit: Maximum number of results (1–100, default 50).
        offset: Result offset (default 0).
        session: Database session.
        user: Authenticated superadmin user.

    Returns:
        PlatformTenantListResponse with paginated tenant list.
    """
    service = _get_tenant_service(session)
    tenants, total = await service.list_tenants_with_status(
        status=status, limit=limit, offset=offset,
    )
    return PlatformTenantListResponse(
        tenants=[
            PlatformTenantResponse(
                id=t.id,
                name=t.name,
                slug=t.slug,
                status=t.status,
                config_json=t.config_json,
                created_at=t.created_at,
            )
            for t in tenants
        ],
        total=total,
    )


@router.patch("/{tenant_id}/status", response_model=PlatformTenantResponse)
async def platform_update_tenant_status(
    tenant_id: str,
    request: TenantStatusUpdateRequest,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_superadmin()),
) -> PlatformTenantResponse:
    """Update a tenant's lifecycle status (superadmin only).

    Valid transitions:
        - pending → active
        - active → suspended, archived
        - suspended → active, archived
        - archived → (terminal, no transitions out)

    Args:
        tenant_id: Tenant UUID.
        request: New status value.
        session: Database session.
        user: Authenticated superadmin user.

    Returns:
        Updated PlatformTenantResponse.
    """
    service = _get_tenant_service(session)
    tenant = await service.update_tenant_status(tenant_id, request.status)
    logger.info(
        "platform_tenant_status_updated",
        tenant_id=tenant_id,
        new_status=request.status.value,
        updated_by=user.id,
    )
    return PlatformTenantResponse(
        id=tenant.id,
        name=tenant.name,
        slug=tenant.slug,
        status=tenant.status,
        config_json=tenant.config_json,
        created_at=tenant.created_at,
    )
