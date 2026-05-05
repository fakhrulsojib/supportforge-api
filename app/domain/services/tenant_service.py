"""Tenant domain service — orchestrates tenant lifecycle operations.

Handles creation, retrieval, update, and deletion of tenants
with proper slug uniqueness validation. ChromaDB collection
management will be integrated in Phase 2.3 (ingestion worker).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from app.core.exceptions import SupportForgeError, TenantNotFoundError

if TYPE_CHECKING:
    from app.domain.interfaces.repository import TenantRepository
    from app.domain.models.tenant import Tenant, TenantCreate

logger = structlog.get_logger(__name__)


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
            raise TenantNotFoundError(tenant_id=slug)
        return tenant

    async def list_tenants(self) -> list[Tenant]:
        """List all tenants.

        Returns:
            List of all Tenant domain models.
        """
        return await self._tenant_repo.list_all()

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
