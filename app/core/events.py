"""Application lifecycle events — startup and shutdown hooks.

Uses FastAPI's lifespan context manager pattern to initialize
and tear down shared resources (DB pool, Redis, ChromaDB, logging).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from fastapi import FastAPI

logger = structlog.get_logger(__name__)


def _configure_structlog(log_level: str) -> None:
    """Configure structlog for JSON output with request-ID correlation."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if log_level == "DEBUG" else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level, logging.INFO),
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Startup:
        - Configure structured logging
        - Log startup message

    Shutdown:
        - Log shutdown message
        - Clean up resources

    DB, Redis, and ChromaDB initialization will be added in
    sub-phases 1.2, 1.4, and 2.7 respectively.
    """
    from app.config import get_settings

    settings = get_settings()

    # ── Startup ──────────────────────────────────────────────────
    _configure_structlog(settings.app_log_level)
    logger.info(
        "starting_application",
        app_name=settings.app_name,
        environment=settings.app_env,
        debug=settings.app_debug,
    )

    # Initialize Redis cache adapter
    import redis.asyncio as aioredis

    from app.infrastructure.cache.redis_adapter import RedisAdapter

    try:
        redis_client = aioredis.from_url(
            settings.computed_redis_url,
            decode_responses=False,
            socket_connect_timeout=5,
        )
        await redis_client.ping()
        app.state.cache = RedisAdapter(redis_client)
        logger.info("redis_connected", url=settings.computed_redis_url)
    except Exception:
        logger.warning("redis_connection_failed", exc_info=True)
        # Create a no-op cache that always returns None
        app.state.cache = None

    yield

    # ── Shutdown ─────────────────────────────────────────────────
    if getattr(app.state, "cache", None) is not None:
        await app.state.cache.close()
        logger.info("redis_disconnected")
    logger.info("shutting_down_application")
