"""SQLAlchemy implementation of TenantRepository."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from app.domain.interfaces.repository import TenantRepository
from app.domain.models.tenant import Tenant, TenantCreate
from app.infrastructure.database.models import TenantModel

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class SQLTenantRepository(TenantRepository):
    """Concrete tenant repository backed by PostgreSQL via SQLAlchemy."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def _to_domain(self, model: TenantModel) -> Tenant:
        """Convert ORM model to domain model."""
        return Tenant(
            id=model.id,
            name=model.name,
            slug=model.slug,
            config_json=model.config_json,
            created_at=model.created_at,
        )

    async def create(self, tenant: TenantCreate) -> Tenant:
        """Create a new tenant."""
        model = TenantModel(
            name=tenant.name,
            slug=tenant.slug,
            config_json=tenant.config_json,
        )
        self._session.add(model)
        await self._session.flush()
        return self._to_domain(model)

    async def get_by_id(self, tenant_id: str) -> Tenant | None:
        """Get a tenant by ID."""
        result = await self._session.get(TenantModel, tenant_id)
        return self._to_domain(result) if result else None

    async def get_by_slug(self, slug: str) -> Tenant | None:
        """Get a tenant by slug."""
        stmt = select(TenantModel).where(TenantModel.slug == slug)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_domain(model) if model else None

    async def list_all(self) -> list[Tenant]:
        """List all tenants."""
        stmt = select(TenantModel).order_by(TenantModel.created_at)
        result = await self._session.execute(stmt)
        return [self._to_domain(m) for m in result.scalars().all()]

    async def update(self, tenant_id: str, **kwargs: Any) -> Tenant | None:
        """Update a tenant's fields."""
        model = await self._session.get(TenantModel, tenant_id)
        if not model:
            return None
        for key, value in kwargs.items():
            if hasattr(model, key):
                setattr(model, key, value)
        await self._session.flush()
        return self._to_domain(model)

    async def delete(self, tenant_id: str) -> bool:
        """Delete a tenant by ID."""
        model = await self._session.get(TenantModel, tenant_id)
        if not model:
            return False
        await self._session.delete(model)
        await self._session.flush()
        return True
