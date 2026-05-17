"""Bootstrap script — creates database tables for a fresh installation.

The superadmin user and management tenant are auto-created by the app
on startup (see app/core/events.py). This script only needs to run once
to initialize the schema before the first app launch.

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


async def bootstrap() -> None:
    """Create all database tables.

    Reads DATABASE_URL from .env via app settings.
    Idempotent: existing tables are not modified.
    """
    from sqlalchemy.ext.asyncio import create_async_engine

    from app.infrastructure.database.models import Base

    settings = get_settings()
    engine = create_async_engine(settings.computed_database_url, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await engine.dispose()

    print("✅ Database tables created")  # noqa: T201
    print(f"   DB: {settings.computed_database_url.split('@')[-1]}")  # noqa: T201
    print("\n   Start the app to auto-create the superadmin user.")  # noqa: T201
    print("   (Requires SUPERADMIN_EMAIL and SUPERADMIN_PASSWORD in .env)")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(bootstrap())
