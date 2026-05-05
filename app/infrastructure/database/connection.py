"""Async SQLAlchemy database connection management.

Provides the async engine, session factory, and session generator
for dependency injection.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings


def create_engine(database_url: str | None = None) -> create_async_engine:  # type: ignore[valid-type]
    """Create an async SQLAlchemy engine.

    Args:
        database_url: Override URL for testing. Falls back to settings.
    """
    if database_url is None:
        settings = get_settings()
        database_url = settings.computed_database_url

    return create_async_engine(
        database_url,
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )


# Default engine — initialized at module load
_engine = create_engine()

# Session factory
AsyncSessionLocal = async_sessionmaker(
    bind=_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session.

    Used as a FastAPI dependency via ``Depends(get_async_session)``.
    The session is automatically closed when the request completes.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
