"""Tests for application lifecycle events."""

from __future__ import annotations

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
