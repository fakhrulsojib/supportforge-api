"""Redis cache adapter — concrete implementation of CachePort.

Wraps the redis.asyncio client with graceful fallback on connection
failure, structured logging, and atomic operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from app.domain.interfaces.cache import CachePort

if TYPE_CHECKING:
    import redis.asyncio as aioredis

logger = structlog.get_logger(__name__)


class RedisAdapter(CachePort):
    """Redis-backed cache adapter.

    Falls back to returning None / no-ops when Redis is unavailable
    to prevent cache failures from breaking the application.

    Attributes:
        _client: Async Redis client instance.
    """

    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client

    async def get(self, key: str) -> str | None:
        """Get a value from Redis by key.

        Returns None if the key doesn't exist or Redis is unreachable.
        """
        try:
            value = await self._client.get(key)
            if value is None:
                return None
            return str(value)
        except Exception:
            logger.warning("redis_get_failed", key=key, exc_info=True)
            return None

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        """Set a value in Redis with optional TTL.

        Silently fails if Redis is unreachable.
        """
        try:
            if ttl_seconds is not None:
                await self._client.setex(key, ttl_seconds, value)
            else:
                await self._client.set(key, value)
        except Exception:
            logger.warning("redis_set_failed", key=key, exc_info=True)

    async def delete(self, key: str) -> None:
        """Delete a key from Redis.

        Silently fails if Redis is unreachable.
        """
        try:
            await self._client.delete(key)
        except Exception:
            logger.warning("redis_delete_failed", key=key, exc_info=True)

    async def incr(self, key: str) -> int:
        """Atomically increment a Redis counter.

        Returns 0 on Redis failure (safe fallback for rate limiting).
        """
        try:
            return await self._client.incr(key)  # type: ignore[no-any-return]
        except Exception:
            logger.warning("redis_incr_failed", key=key, exc_info=True)
            return 0

    async def close(self) -> None:
        """Close the Redis connection."""
        try:
            await self._client.aclose()
        except Exception:
            logger.warning("redis_close_failed", exc_info=True)
