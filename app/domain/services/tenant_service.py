"""Tenant domain service — orchestrates tenant lifecycle operations.

Handles creation, retrieval, update, and deletion of tenants
with proper slug uniqueness validation. ChromaDB collection
management will be integrated in Phase 2.3 (ingestion worker).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from app.core.exceptions import SupportForgeError, TenantNotFoundError
from app.domain.models.enums import TenantStatus

if TYPE_CHECKING:
    from app.domain.interfaces.repository import TenantRepository
    from app.domain.models.tenant import Tenant, TenantCreate

logger = structlog.get_logger(__name__)

# ── Tenant status transition rules ──────────────────────────────
# Defines which status values can transition to which other values.
# `archived` is terminal — no transitions out.
VALID_TRANSITIONS: dict[TenantStatus, set[TenantStatus]] = {
    TenantStatus.PENDING: {TenantStatus.ACTIVE},
    TenantStatus.ACTIVE: {TenantStatus.SUSPENDED, TenantStatus.ARCHIVED},
    TenantStatus.SUSPENDED: {TenantStatus.ACTIVE, TenantStatus.ARCHIVED},
    TenantStatus.ARCHIVED: set(),  # terminal
}


class TenantService:
    """Domain service for tenant lifecycle management.

    Attributes:
        _tenant_repo: Repository for tenant persistence.
    """

    def __init__(self, tenant_repo: TenantRepository) -> None:
        self._tenant_repo = tenant_repo

    async def create_tenant(self, data: TenantCreate) -> Tenant:
        """Create a new tenant after validating slug uniqueness.

        Args:
            data: Tenant creation data (name, slug, optional config_json).

        Returns:
            The newly created Tenant domain model.

        Raises:
            SupportForgeError(409): If a tenant with the same slug already exists.
        """
        existing = await self._tenant_repo.get_by_slug(data.slug)
        if existing:
            raise SupportForgeError(
                message=f"Tenant with slug '{data.slug}' already exists",
                status_code=409,
                error_code="TENANT_SLUG_EXISTS",
            )

        tenant = await self._tenant_repo.create(data)
        logger.info("tenant_created", tenant_id=tenant.id, slug=tenant.slug)
        return tenant

    async def get_tenant_by_id(self, tenant_id: str) -> Tenant:
        """Retrieve a tenant by ID.

        Args:
            tenant_id: Tenant UUID string.

        Returns:
            The Tenant domain model.

        Raises:
            TenantNotFoundError: If no tenant exists with the given ID.
        """
        tenant = await self._tenant_repo.get_by_id(tenant_id)
        if not tenant:
            raise TenantNotFoundError(tenant_id=tenant_id)
        return tenant

    async def get_tenant_by_slug(self, slug: str) -> Tenant:
        """Retrieve a tenant by slug.

        Args:
            slug: Tenant slug string.

        Returns:
            The Tenant domain model.

        Raises:
            TenantNotFoundError: If no tenant exists with the given slug.
        """
        tenant = await self._tenant_repo.get_by_slug(slug)
        if not tenant:
            raise TenantNotFoundError(tenant_id=f"slug:{slug}")
        return tenant

    async def list_tenants(self) -> list[Tenant]:
        """List all tenants.

        Returns:
            List of all Tenant domain models.
        """
        return await self._tenant_repo.list_all()

    async def list_tenants_with_status(
        self,
        *,
        status: TenantStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Tenant], int]:
        """List tenants with optional status filter and pagination.

        Args:
            status: Optional status filter.
            limit: Maximum number of results.
            offset: Result offset.

        Returns:
            Tuple of (tenants list, total count).
        """
        tenants = await self._tenant_repo.list_all_with_status(
            status=status, limit=limit, offset=offset,
        )
        total = await self._tenant_repo.count_all(status=status)
        return tenants, total

    async def update_tenant(self, tenant_id: str, **kwargs: Any) -> Tenant:
        """Update a tenant's mutable fields.

        Args:
            tenant_id: Tenant UUID string.
            **kwargs: Fields to update (name, config_json).

        Returns:
            Updated Tenant domain model.

        Raises:
            TenantNotFoundError: If no tenant exists with the given ID.
        """
        tenant = await self._tenant_repo.update(tenant_id, **kwargs)
        if not tenant:
            raise TenantNotFoundError(tenant_id=tenant_id)
        logger.info("tenant_updated", tenant_id=tenant_id, fields=list(kwargs.keys()))
        return tenant

    async def update_tenant_status(
        self, tenant_id: str, new_status: TenantStatus,
    ) -> Tenant:
        """Validate and execute a tenant status transition.

        Args:
            tenant_id: Tenant UUID string.
            new_status: Target status to transition to.

        Returns:
            Updated Tenant domain model.

        Raises:
            TenantNotFoundError: If no tenant exists with the given ID.
            SupportForgeError(400): If the transition is invalid.
        """
        tenant = await self._tenant_repo.get_by_id(tenant_id)
        if not tenant:
            raise TenantNotFoundError(tenant_id=tenant_id)

        current = tenant.status
        allowed = VALID_TRANSITIONS.get(current, set())

        if new_status not in allowed:
            raise SupportForgeError(
                message=(
                    f"Invalid status transition: '{current.value}' → '{new_status.value}'. "
                    f"Allowed transitions from '{current.value}': "
                    f"{', '.join(s.value for s in sorted(allowed, key=lambda x: x.value)) or 'none (terminal state)'}"
                ),
                status_code=400,
                error_code="INVALID_STATUS_TRANSITION",
            )

        updated = await self._tenant_repo.update_status(tenant_id, new_status)
        if not updated:
            raise TenantNotFoundError(tenant_id=tenant_id)

        logger.info(
            "tenant_status_updated",
            tenant_id=tenant_id,
            from_status=current.value,
            to_status=new_status.value,
        )
        return updated

    async def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant.

        Args:
            tenant_id: Tenant UUID string.

        Returns:
            True if deleted.

        Raises:
            TenantNotFoundError: If no tenant exists with the given ID.
        """
        deleted = await self._tenant_repo.delete(tenant_id)
        if not deleted:
            raise TenantNotFoundError(tenant_id=tenant_id)
        logger.info("tenant_deleted", tenant_id=tenant_id)
        return True
