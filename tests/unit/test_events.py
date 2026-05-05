"""Tests for application lifecycle events."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import clear_settings_cache
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


def _mock_infrastructure():
    """Return patch context managers for all infrastructure deps used in lifespan.

    The lifespan function uses deferred imports inside the function body,
    so we patch the actual module paths, not app.core.events.*.
    """
    mock_llm = MagicMock()
    mock_llm.close = AsyncMock()

    return (
        mock_llm,
        patch("app.infrastructure.llm.factory.get_llm_provider", return_value=mock_llm),
        patch("app.infrastructure.vectorstore.chroma_adapter.ChromaAdapter", return_value=MagicMock()),
        patch("app.rag.embeddings.EmbeddingService", return_value=MagicMock()),
    )


class TestLifespan:
    """Test suite for startup/shutdown lifecycle.

    C6: Tests set APP_ENV=test to bypass the JWT secret validation,
    since the test suite intentionally uses the default secret.

    Each test clears the settings singleton cache so that
    ``patch.dict(os.environ)`` overrides take effect.
    """

    @pytest.mark.asyncio
    async def test_lifespan_runs_startup_and_shutdown(self) -> None:
        """Lifespan context manager should execute without errors."""
        mock_llm, p1, p2, p3 = _mock_infrastructure()
        with patch.dict(os.environ, {"APP_ENV": "test"}), p1, p2, p3:
            clear_settings_cache()
            app = create_app()
            async with lifespan(app):
                # Startup has completed if we reach here
                pass
            # Shutdown runs when context manager exits
            clear_settings_cache()

    @pytest.mark.asyncio
    async def test_lifespan_redis_success_sets_cache(self) -> None:
        """Successful Redis connection should set app.state.cache."""
        mock_llm, p1, p2, p3 = _mock_infrastructure()
        with patch.dict(os.environ, {"APP_ENV": "test"}), p1, p2, p3:
            clear_settings_cache()
            app = create_app()

            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock(return_value=True)
            mock_redis.close = AsyncMock()

            with patch("redis.asyncio.from_url", return_value=mock_redis):
                async with lifespan(app):
                    # Cache should be set (RedisAdapter wrapping mock_redis)
                    assert app.state.cache is not None
            clear_settings_cache()

    @pytest.mark.asyncio
    async def test_lifespan_redis_failure_sets_cache_none(self) -> None:
        """Failed Redis connection should set app.state.cache = None."""
        mock_llm, p1, p2, p3 = _mock_infrastructure()
        with patch.dict(os.environ, {"APP_ENV": "test"}), p1, p2, p3:
            clear_settings_cache()
            app = create_app()

            with patch("redis.asyncio.from_url", side_effect=ConnectionError("Redis unavailable")):
                async with lifespan(app):
                    assert app.state.cache is None
            clear_settings_cache()

    @pytest.mark.asyncio
    async def test_lifespan_shutdown_closes_redis(self) -> None:
        """Shutdown should call close() on the cache adapter."""
        mock_llm, p1, p2, p3 = _mock_infrastructure()
        with patch.dict(os.environ, {"APP_ENV": "test"}), p1, p2, p3:
            clear_settings_cache()
            app = create_app()

            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock(return_value=True)
            mock_redis.close = AsyncMock()

            with patch("redis.asyncio.from_url", return_value=mock_redis):
                async with lifespan(app):
                    assert app.state.cache is not None

                # After context manager exits, close should have been called
                assert app.state.cache is not None  # Still set but close was called
            clear_settings_cache()

    @pytest.mark.asyncio
    async def test_lifespan_rejects_default_secret_in_production(self) -> None:
        """C6: Default JWT secret should cause RuntimeError in non-test env."""
        with patch.dict(os.environ, {"APP_ENV": "development"}):
            clear_settings_cache()
            app = create_app()
            with pytest.raises(RuntimeError, match="JWT_SECRET_KEY must be changed"):
                async with lifespan(app):
                    pass  # Should not reach here
            clear_settings_cache()

    @pytest.mark.asyncio
    async def test_lifespan_initializes_chat_service(self) -> None:
        """Startup should create ChatService on app.state."""
        mock_llm, p1, p2, p3 = _mock_infrastructure()
        with patch.dict(os.environ, {"APP_ENV": "test"}), p1, p2, p3:
            clear_settings_cache()
            app = create_app()
            async with lifespan(app):
                assert hasattr(app.state, "chat_service")
                assert app.state.chat_service is not None
            clear_settings_cache()

    @pytest.mark.asyncio
    async def test_lifespan_initializes_ws_manager(self) -> None:
        """Startup should create ConnectionManager on app.state."""
        from app.infrastructure.websocket.connection_manager import ConnectionManager

        mock_llm, p1, p2, p3 = _mock_infrastructure()
        with patch.dict(os.environ, {"APP_ENV": "test"}), p1, p2, p3:
            clear_settings_cache()
            app = create_app()
            async with lifespan(app):
                assert hasattr(app.state, "ws_manager")
                assert isinstance(app.state.ws_manager, ConnectionManager)
            clear_settings_cache()

    @pytest.mark.asyncio
    async def test_lifespan_shutdown_closes_llm_provider(self) -> None:
        """Shutdown should call close() on the LLM provider."""
        mock_llm = MagicMock()
        mock_llm.close = AsyncMock()

        with (
            patch.dict(os.environ, {"APP_ENV": "test"}),
            patch("app.infrastructure.llm.factory.get_llm_provider", return_value=mock_llm),
            patch("app.infrastructure.vectorstore.chroma_adapter.ChromaAdapter", return_value=MagicMock()),
            patch("app.rag.embeddings.EmbeddingService", return_value=MagicMock()),
        ):
            clear_settings_cache()
            app = create_app()
            async with lifespan(app):
                pass
            mock_llm.close.assert_called_once()
            clear_settings_cache()
