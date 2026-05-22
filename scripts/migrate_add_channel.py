"""Add 'channel' column to the messages table.

This migration adds the MessageChannel enum type and the 'channel'
column for existing databases. New databases get it via create_all().

Usage:
    python scripts/migrate_add_channel.py

Idempotent: safe to run multiple times.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.config import get_settings  # noqa: E402


async def migrate() -> None:
    """Add 'channel' column and index to the messages table."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    settings = get_settings()
    engine = create_async_engine(settings.computed_database_url, echo=False)

    async with engine.begin() as conn:
        # Create the enum type if it doesn't exist
        await conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'messagechannel') THEN
                    CREATE TYPE messagechannel AS ENUM ('text', 'voice');
                END IF;
            END
            $$;
        """))

        # Add the column if it doesn't exist
        await conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'messages' AND column_name = 'channel'
                ) THEN
                    ALTER TABLE messages
                        ADD COLUMN channel messagechannel NOT NULL DEFAULT 'text';
                END IF;
            END
            $$;
        """))

        # Add the index if it doesn't exist
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_messages_channel ON messages (channel);
        """))

    await engine.dispose()

    print("✅ Migration complete: 'channel' column added to messages table")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(migrate())
