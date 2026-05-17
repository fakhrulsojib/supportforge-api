"""Bootstrap script — creates tables, platform tenant, and superadmin user.

Self-sufficient: reads all config from .env (SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD).
Auto-creates a "Platform" system tenant for the superadmin if one doesn't exist.
Idempotent: safe to run multiple times.

Usage:
    python scripts/create_superadmin.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `app` can be imported.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.config import get_settings  # noqa: E402
from app.core.security import hash_password  # noqa: E402
from app.domain.models.enums import UserRole  # noqa: E402
from app.domain.models.tenant import TenantCreate  # noqa: E402
from app.domain.models.user import UserCreate  # noqa: E402

# Platform tenant defaults
PLATFORM_TENANT_NAME = "Platform"
PLATFORM_TENANT_SLUG = "platform"


async def bootstrap() -> None:
    """Create tables, platform tenant, and superadmin user.

    All config is read from .env via app settings:
    - DATABASE_URL: PostgreSQL connection string
    - SUPERADMIN_EMAIL: superadmin login email
    - SUPERADMIN_PASSWORD: superadmin login password
    """
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from app.infrastructure.database.models import Base
    from app.infrastructure.database.repositories.tenant_repo import SQLTenantRepository
    from app.infrastructure.database.repositories.user_repo import SQLUserRepository

    settings = get_settings()

    email = settings.superadmin_email
    password = settings.superadmin_password

    if not email or not password:
        print("❌ SUPERADMIN_EMAIL and SUPERADMIN_PASSWORD must be set in .env")  # noqa: T201
        sys.exit(1)

    engine = create_async_engine(settings.computed_database_url, echo=False)

    # 1. Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database tables created")  # noqa: T201

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        # 2. Create or find platform tenant
        tenant_repo = SQLTenantRepository(session)
        tenant = await tenant_repo.get_by_slug(PLATFORM_TENANT_SLUG)
        if tenant:
            print(f"  ⚠ Platform tenant already exists: {tenant.id}")  # noqa: T201
        else:
            tenant = await tenant_repo.create(
                TenantCreate(name=PLATFORM_TENANT_NAME, slug=PLATFORM_TENANT_SLUG),
            )
            print(f"  ✅ Platform tenant created: {tenant.id}")  # noqa: T201

        # 3. Create or find superadmin user
        user_repo = SQLUserRepository(session)
        existing = await user_repo.get_by_email(email, tenant.id)
        if existing:
            print(f"  ⚠ Superadmin '{email}' already exists (role: {existing.role.value})")  # noqa: T201
        else:
            hashed = hash_password(password)
            user = await user_repo.create(
                tenant.id,
                UserCreate(email=email, role=UserRole.SUPERADMIN),
                password_hash=hashed,
            )
            print(f"  ✅ Superadmin created: {user.id}")  # noqa: T201

        await session.commit()

    print(f"\n  Tenant ID: {tenant.id}")  # noqa: T201
    print(f"  Email:     {email}")  # noqa: T201
    print(f"  Role:      superadmin")  # noqa: T201
    print(f"  DB:        {settings.computed_database_url.split('@')[-1]}")  # noqa: T201

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(bootstrap())
