"""Shared test fixtures for SupportForge API tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@pytest.fixture(autouse=True)
def _reset_settings_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure every test uses the default JWT secret, not the .env override.

    All test token fixtures create JWTs signed with the Settings class
    default ``change-me-to-another-random-secret``.  The ``.env`` file
    may override ``JWT_SECRET_KEY`` with a different value, causing
    ``verify_token`` to reject test tokens with a 401.

    This autouse fixture:
    1. Sets ``JWT_SECRET_KEY`` in the environment to the default.
    2. Clears the cached Settings singleton so it picks up the env var.
    """
    monkeypatch.setenv("JWT_SECRET_KEY", "change-me-to-another-random-secret")
    from app.config import clear_settings_cache
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture
def app():
    """Create a fresh FastAPI application for each test."""
    return create_app()


@pytest.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP test client for the FastAPI application."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac

