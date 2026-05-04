"""Create a demo tenant via the API or directly in the database.

Usage:
    python scripts/create_tenant.py [--name NAME] [--slug SLUG]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402


async def create_tenant(name: str, slug: str) -> None:
    """Create a demo tenant in the database.

    This is a simplified script for bootstrapping. In production,
    tenants are created via the Admin API (Phase 3).
    """
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from app.domain.models.tenant import TenantCreate
    from app.infrastructure.database.models import Base
    from app.infrastructure.database.repositories.tenant_repo import SQLTenantRepository

    settings = get_settings()
    engine = create_async_engine(settings.computed_database_url, echo=True)

    # Create tables if they don't exist (dev only)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        repo = SQLTenantRepository(session)

        # Check if tenant already exists (idempotent)
        existing = await repo.get_by_slug(slug)
        if existing:
            print(f"✓ Tenant '{name}' (slug: {slug}) already exists with ID: {existing.id}")
            return

        tenant_data = TenantCreate(name=name, slug=slug)
        tenant = await repo.create(tenant_data)
        await session.commit()
        print(f"✓ Created tenant '{tenant.name}' with ID: {tenant.id}")

    await engine.dispose()


def main() -> None:
    """Parse CLI args and create tenant."""
    parser = argparse.ArgumentParser(description="Create a demo tenant")
    parser.add_argument("--name", default="Demo Support Center", help="Tenant name")
    parser.add_argument("--slug", default="demo-support", help="Tenant slug")
    args = parser.parse_args()

    asyncio.run(create_tenant(args.name, args.slug))


if __name__ == "__main__":
    main()
