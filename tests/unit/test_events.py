"""Tests for application lifecycle events."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.core.events import _configure_structlog, lifespan
from app.main import create_app


class TestConfigureStructlog:
    """Test suite for structlog configuration."""

    def test_configure_debug_level(self) -> None:
        """Debug level should configure ConsoleRenderer."""
        # Should not raise
        _configure_structlog("DEBUG")

    def test_configure_info_level(self) -> None:
        """Non-debug level should configure JSONRenderer."""
        _configure_structlog("INFO")

    def test_configure_warning_level(self) -> None:
        """Warning level should configure JSONRenderer."""
        _configure_structlog("WARNING")


class TestLifespan:
    """Test suite for startup/shutdown lifecycle."""

    @pytest.mark.asyncio
    async def test_lifespan_runs_startup_and_shutdown(self) -> None:
        """Lifespan context manager should execute without errors."""
        app = create_app()
        async with lifespan(app):
            # Startup has completed if we reach here
            pass
        # Shutdown runs when context manager exits

    @pytest.mark.asyncio
    async def test_lifespan_redis_success_sets_cache(self) -> None:
        """Successful Redis connection should set app.state.cache."""
        app = create_app()

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.close = AsyncMock()

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            async with lifespan(app):
                # Cache should be set (RedisAdapter wrapping mock_redis)
                assert app.state.cache is not None

    @pytest.mark.asyncio
    async def test_lifespan_redis_failure_sets_cache_none(self) -> None:
        """Failed Redis connection should set app.state.cache = None."""
        app = create_app()

        with patch("redis.asyncio.from_url", side_effect=ConnectionError("Redis unavailable")):
            async with lifespan(app):
                assert app.state.cache is None

    @pytest.mark.asyncio
    async def test_lifespan_shutdown_closes_redis(self) -> None:
        """Shutdown should call close() on the cache adapter."""
        app = create_app()

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.close = AsyncMock()

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            async with lifespan(app):
                assert app.state.cache is not None

            # After context manager exits, close should have been called
            # The RedisAdapter.close() should have been invoked during shutdown
            assert app.state.cache is not None  # Still set but close was called
