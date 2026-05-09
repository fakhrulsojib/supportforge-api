"""Bootstrap script — creates tenant + admin user directly in DB.

Bypasses the API auth chicken-and-egg problem when DB is freshly cleared.
"""

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.security import hash_password
from app.domain.models.enums import UserRole
from app.domain.models.tenant import TenantCreate
from app.domain.models.user import UserCreate
from app.infrastructure.database.models import Base
from app.infrastructure.database.repositories.tenant_repo import SQLTenantRepository
from app.infrastructure.database.repositories.user_repo import SQLUserRepository

DATABASE_URL = "postgresql+asyncpg://supportforge:REDACTED_DB_PASSWORD@localhost:5433/supportforge"


async def bootstrap():
    engine = create_async_engine(DATABASE_URL, echo=False)

    # Create tables if not exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # 1. Create tenant
        tenant_repo = SQLTenantRepository(session)
        existing = await tenant_repo.get_by_slug("novamart")
        if existing:
            tenant = existing
            print(f"  ⚠ Tenant 'novamart' already exists: {tenant.id}")
        else:
            tenant = await tenant_repo.create(TenantCreate(name="NovaMart", slug="novamart"))
            print(f"  ✅ Tenant created: {tenant.id}")

        # 2. Create admin user
        user_repo = SQLUserRepository(session)
        existing_user = await user_repo.get_by_email("admin@novamart.com", tenant.id)
        if existing_user:
            print(f"  ⚠ Admin already exists: {existing_user.id}")
        else:
            hashed = hash_password("NovaMart2025!@#")
            user = await user_repo.create(
                tenant.id,
                UserCreate(email="admin@novamart.com", role=UserRole.ADMIN),
                password_hash=hashed,
            )
            print(f"  ✅ Admin created: {user.id}")

        await session.commit()

    print(f"\n  Tenant ID: {tenant.id}")
    print("  Email:     admin@novamart.com")
    print("  Password:  NovaMart2025!@#")
    print("  Role:      admin")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(bootstrap())
