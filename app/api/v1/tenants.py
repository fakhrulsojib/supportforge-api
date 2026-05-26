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
    TestHookRequest,
    TestHookResponse,
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


@router.post("", response_model=TenantResponse, status_code=201, deprecated=True)
async def create_tenant(
    request: TenantCreateRequest,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> TenantResponse:
    """Create a new tenant (admin only).

    .. deprecated::
        Use ``POST /api/v1/platform/tenants`` (superadmin-only) instead.
        This endpoint will be removed in a future release.

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


@router.get("", response_model=TenantListResponse)
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
    if user.role != UserRole.SUPERADMIN and user.tenant_id != tenant_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Not authorized to modify this tenant")

    service = _get_tenant_service(session)
    update_data = request.model_dump(exclude_unset=True)

    # Validate config_json structure if provided
    if "config_json" in update_data and update_data["config_json"]:
        from app.core.config_validators import validate_config_json

        update_data["config_json"] = validate_config_json(update_data["config_json"])

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
    if user.role != UserRole.SUPERADMIN and user.tenant_id != tenant_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Not authorized to modify this tenant")

    service = _get_tenant_service(session)
    await service.delete_tenant(tenant_id)
    logger.info("tenant_deleted_via_api", tenant_id=tenant_id, deleted_by=user.id)


@router.post("/{tenant_id}/test-hook", response_model=TestHookResponse)
async def test_event_hook(
    tenant_id: str,
    request: TestHookRequest,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> TestHookResponse:
    """Test a webhook URL by sending a sample payload.

    Sends an HTTP POST with a sample event payload to verify the
    tenant's webhook URL is reachable and responding.  Uses SSRF
    protection to prevent internal network access.

    Args:
        tenant_id: Tenant UUID (must match authenticated user's tenant).
        request: Test hook configuration (event_type, url, headers).
        session: Database session.
        user: Authenticated admin user.

    Returns:
        TestHookResponse with success status and HTTP status code.
    """
    # Verify tenant ownership
    if user.tenant_id != tenant_id and user.role != UserRole.SUPERADMIN:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="You can only test hooks for your own tenant")

    try:
        import httpx
        from urllib.parse import urlparse

        from app.rag.tools.webhook import validate_url_safety

        # SSRF protection: resolve safe IP
        url_str = str(request.url)
        safe_ip = await validate_url_safety(url_str)

        parsed = urlparse(url_str)
        original_hostname = parsed.hostname
        send_headers = {
            "Content-Type": "application/json",
            "User-Agent": "SupportForge-Hooks/1.0",
            **request.headers,
        }
        if original_hostname:
            send_headers["Host"] = original_hostname
            port_part = f":{parsed.port}" if parsed.port else ""
            # Strip auth credentials and rebuild safely
            safe_url = f"{parsed.scheme}://{safe_ip}{port_part}{parsed.path}"
            if parsed.query:
                safe_url += f"?{parsed.query}"
        else:
            safe_url = url_str

        # Build sample payload
        from datetime import datetime, timezone

        sample_payload = {
            "event": request.event_type,
            "tenant_id": tenant_id,
            "conversation_id": "test-00000000-0000-0000-0000-000000000000",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"test": True, "message": "This is a test webhook from SupportForge"},
        }

        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            response = await client.post(str(safe_url), json=sample_payload, headers=send_headers)

        logger.info(
            "test_hook_dispatched",
            tenant_id=tenant_id,
            event_type=request.event_type,
            url=request.url,
            status=response.status_code,
        )

        return TestHookResponse(
            success=200 <= response.status_code < 300,
            status_code=response.status_code,
            error=None if 200 <= response.status_code < 300 else f"HTTP {response.status_code}",
        )

    except httpx.TimeoutException:
        return TestHookResponse(success=False, error="Request timed out (10s)")
    except Exception as exc:
        logger.warning(
            "test_hook_failed",
            tenant_id=tenant_id,
            url=str(request.url),
            exc_info=False,
        )
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(exc)[:200])

