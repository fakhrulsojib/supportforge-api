"""Abstract cache interface — hexagonal port for caching.

Defines the contract that cache adapters (Redis, in-memory, etc.)
must implement. Used throughout the domain layer for session context,
rate limiting, and token management without coupling to Redis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class CachePort(ABC):
    """Abstract cache interface.

    All cache operations are async and work with string key-value pairs.
    Implementations should handle serialization/deserialization
    of complex types at the adapter level.
    """

    @abstractmethod
    async def get(self, key: str) -> str | None:
        """Retrieve a value by key.

        Args:
            key: Cache key.

        Returns:
            Cached string value, or None if not found / expired.
        """

    @abstractmethod
    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        """Store a value with optional TTL.

        Args:
            key: Cache key.
            value: String value to store.
            ttl_seconds: Time-to-live in seconds. None = no expiry.
        """

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a key from the cache.

        Args:
            key: Cache key to delete.
        """

    @abstractmethod
    async def incr(self, key: str) -> int:
        """Atomically increment a counter.

        Creates the key with value 1 if it doesn't exist.

        Args:
            key: Counter key.

        Returns:
            New counter value after increment.
        """

    @abstractmethod
    async def close(self) -> None:
        """Close the underlying connection.

        Called during application shutdown to release resources.
        Implementations should handle already-closed connections
        gracefully (no-op or log a warning).
        """
