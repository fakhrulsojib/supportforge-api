"""Unit tests for Redis cache adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.infrastructure.cache.redis_adapter import RedisAdapter


@pytest.fixture
def mock_redis() -> AsyncMock:
    """Create a mock async Redis client."""
    return AsyncMock()


@pytest.fixture
def adapter(mock_redis: AsyncMock) -> RedisAdapter:
    """Create RedisAdapter with mocked client."""
    return RedisAdapter(client=mock_redis)


class TestRedisGet:
    """Tests for RedisAdapter.get."""

    async def test_get_hit_bytes(self, adapter: RedisAdapter, mock_redis: AsyncMock) -> None:
        """Should decode bytes and return string."""
        mock_redis.get.return_value = b"cached-value"
        result = await adapter.get("key1")
        assert result == "cached-value"
        mock_redis.get.assert_awaited_once_with("key1")

    async def test_get_hit_string(self, adapter: RedisAdapter, mock_redis: AsyncMock) -> None:
        """Should handle string return directly."""
        mock_redis.get.return_value = "string-value"
        result = await adapter.get("key2")
        assert result == "string-value"

    async def test_get_miss(self, adapter: RedisAdapter, mock_redis: AsyncMock) -> None:
        """Should return None for cache miss."""
        mock_redis.get.return_value = None
        result = await adapter.get("missing")
        assert result is None

    async def test_get_failure_returns_none(self, adapter: RedisAdapter, mock_redis: AsyncMock) -> None:
        """Should return None when Redis connection fails."""
        mock_redis.get.side_effect = ConnectionError("Redis down")
        result = await adapter.get("failing")
        assert result is None


class TestRedisSet:
    """Tests for RedisAdapter.set."""

    async def test_set_without_ttl(self, adapter: RedisAdapter, mock_redis: AsyncMock) -> None:
        """Should call set without TTL."""
        await adapter.set("key", "value")
        mock_redis.set.assert_awaited_once_with("key", "value")

    async def test_set_with_ttl(self, adapter: RedisAdapter, mock_redis: AsyncMock) -> None:
        """Should call setex with TTL."""
        await adapter.set("key", "value", ttl_seconds=300)
        mock_redis.setex.assert_awaited_once_with("key", 300, "value")

    async def test_set_failure_silenced(self, adapter: RedisAdapter, mock_redis: AsyncMock) -> None:
        """Should not raise on Redis failure."""
        mock_redis.set.side_effect = ConnectionError("Redis down")
        await adapter.set("key", "value")  # Should not raise


class TestRedisDelete:
    """Tests for RedisAdapter.delete."""

    async def test_delete_success(self, adapter: RedisAdapter, mock_redis: AsyncMock) -> None:
        """Should call delete on Redis."""
        await adapter.delete("key")
        mock_redis.delete.assert_awaited_once_with("key")

    async def test_delete_failure_silenced(self, adapter: RedisAdapter, mock_redis: AsyncMock) -> None:
        """Should not raise on Redis failure."""
        mock_redis.delete.side_effect = ConnectionError("Redis down")
        await adapter.delete("key")  # Should not raise


class TestRedisIncr:
    """Tests for RedisAdapter.incr."""

    async def test_incr_success(self, adapter: RedisAdapter, mock_redis: AsyncMock) -> None:
        """Should return incremented value."""
        mock_redis.incr.return_value = 5
        result = await adapter.incr("counter")
        assert result == 5
        mock_redis.incr.assert_awaited_once_with("counter")

    async def test_incr_failure_returns_zero(self, adapter: RedisAdapter, mock_redis: AsyncMock) -> None:
        """Should return 0 on Redis failure (safe for rate limiting)."""
        mock_redis.incr.side_effect = ConnectionError("Redis down")
        result = await adapter.incr("counter")
        assert result == 0


class TestRedisClose:
    """Tests for RedisAdapter.close."""

    async def test_close_success(self, adapter: RedisAdapter, mock_redis: AsyncMock) -> None:
        """Should close the Redis connection."""
        await adapter.close()
        mock_redis.aclose.assert_awaited_once()

    async def test_close_failure_silenced(self, adapter: RedisAdapter, mock_redis: AsyncMock) -> None:
        """Should not raise on close failure."""
        mock_redis.aclose.side_effect = ConnectionError("Already closed")
        await adapter.close()  # Should not raise
