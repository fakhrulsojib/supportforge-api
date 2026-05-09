"""Bootstrap script to create the first platform superadmin user.

Usage:
    python scripts/create_superadmin.py --email admin@platform.dev --password 'StrongP@ss1' --tenant-id <UUID>

The script is idempotent: if the email already exists in the specified tenant,
it prints a warning and exits without error.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `app` can be imported.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.config import get_settings  # noqa: E402
from app.core.security import hash_password, validate_password_strength  # noqa: E402
from app.domain.models.enums import UserRole  # noqa: E402
from app.domain.models.user import UserCreate  # noqa: E402
from app.infrastructure.database.connection import get_async_session  # noqa: E402
from app.infrastructure.database.repositories.tenant_repo import SQLTenantRepository  # noqa: E402
from app.infrastructure.database.repositories.user_repo import SQLUserRepository  # noqa: E402


async def create_superadmin(email: str, password: str, tenant_id: str) -> None:
    """Create a superadmin user in the database.

    Args:
        email: Superadmin email address.
        password: Plaintext password (will be hashed).
        tenant_id: UUID of the tenant to register under.
    """
    # Validate password strength
    errors = validate_password_strength(password)
    if errors:
        print(f"❌ Password too weak: {'; '.join(errors)}")  # noqa: T201
        sys.exit(1)

    async for session in get_async_session():
        # Verify tenant exists
        tenant_repo = SQLTenantRepository(session)
        tenant = await tenant_repo.get_by_id(tenant_id)
        if not tenant:
            print(f"❌ Tenant '{tenant_id}' not found. Create the tenant first.")  # noqa: T201
            sys.exit(1)

        # Check idempotency
        user_repo = SQLUserRepository(session)
        existing = await user_repo.get_by_email(email, tenant_id)
        if existing:
            print(f"⚠️  User '{email}' already exists in tenant '{tenant.name}' (role: {existing.role.value}).")  # noqa: T201
            if existing.role == UserRole.SUPERADMIN:
                print("   Already a superadmin. No action needed.")  # noqa: T201
            else:
                print(f"   Current role is '{existing.role.value}'. Update manually if needed.")  # noqa: T201
            return

        # Create superadmin user
        hashed = hash_password(password)
        user_create = UserCreate(email=email, role=UserRole.SUPERADMIN)
        user = await user_repo.create(tenant_id, user_create, password_hash=hashed)
        await session.commit()

        print("✅ Superadmin created successfully!")  # noqa: T201
        print(f"   Email:     {user.email}")  # noqa: T201
        print(f"   User ID:   {user.id}")  # noqa: T201
        print(f"   Tenant:    {tenant.name} ({tenant.id})")  # noqa: T201
        print(f"   Role:      {user.role.value}")  # noqa: T201


def main() -> None:
    """Parse CLI arguments and run the superadmin creation."""
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Create a platform superadmin user for SupportForge.",
    )
    parser.add_argument(
        "--email",
        required=True,
        help="Superadmin email address",
    )
    parser.add_argument(
        "--password",
        required=True,
        help="Superadmin password (must meet strength requirements)",
    )
    parser.add_argument(
        "--tenant-id",
        required=True,
        help="UUID of the tenant to register the superadmin under",
    )

    args = parser.parse_args()

    print(f"🔧 Creating superadmin for {settings.app_name}...")  # noqa: T201
    asyncio.run(create_superadmin(args.email, args.password, args.tenant_id))


if __name__ == "__main__":
    main()
