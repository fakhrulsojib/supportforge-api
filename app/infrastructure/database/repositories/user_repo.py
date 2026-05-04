"""SQLAlchemy implementation of UserRepository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from app.domain.interfaces.repository import UserRepository
from app.domain.models.user import User, UserCreate
from app.infrastructure.database.models import UserModel

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class SQLUserRepository(UserRepository):
    """Concrete user repository backed by PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def _to_domain(self, model: UserModel) -> User:
        """Convert ORM model to domain model."""
        return User(
            id=model.id,
            tenant_id=model.tenant_id,
            email=model.email,
            password_hash=model.password_hash,
            role=model.role,
            created_at=model.created_at,
        )

    async def create(self, tenant_id: str, user: UserCreate) -> User:
        """Create a new user within a tenant."""
        model = UserModel(
            tenant_id=tenant_id,
            email=user.email,
            password_hash="",  # Password hashing handled by auth service (Phase 2.5)
            role=user.role,
        )
        self._session.add(model)
        await self._session.flush()
        return self._to_domain(model)

    async def get_by_id(self, user_id: str) -> User | None:
        """Get a user by ID."""
        result = await self._session.get(UserModel, user_id)
        return self._to_domain(result) if result else None

    async def get_by_email(self, email: str, tenant_id: str) -> User | None:
        """Get a user by email within a tenant."""
        stmt = select(UserModel).where(
            UserModel.email == email,
            UserModel.tenant_id == tenant_id,
        )
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_domain(model) if model else None

    async def list_by_tenant(self, tenant_id: str) -> list[User]:
        """List all users for a tenant."""
        stmt = select(UserModel).where(UserModel.tenant_id == tenant_id).order_by(UserModel.created_at)
        result = await self._session.execute(stmt)
        return [self._to_domain(m) for m in result.scalars().all()]
