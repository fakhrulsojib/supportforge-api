"""Application lifecycle events — startup and shutdown hooks.

Uses FastAPI's lifespan context manager pattern to initialize
and tear down shared resources (DB pool, Redis, ChromaDB, logging).
"""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from fastapi import FastAPI

logger = structlog.get_logger(__name__)

# Default JWT secret — used to detect unchanged secrets at startup
_DEFAULT_JWT_SECRET = "change-me-to-another-random-secret"  # noqa: S105

# Regex to mask passwords in Redis URLs (redis://:password@host → redis://:***@host)
_REDIS_PASSWORD_RE = re.compile(r"(rediss?://):([^@]+)@")


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
        - C6: Validate JWT secret is not the default in production
        - Initialize Redis cache (graceful fallback)
        - Initialize ChatService singleton with LLM, VectorStore, Embedding
        - Initialize WebSocket ConnectionManager

    Shutdown:
        - Clean up Redis connection
        - Clean up LLM adapter (close httpx client)
        - Log shutdown
    """
    from app.config import get_settings
    from app.domain.services.chat_service import ChatService
    from app.infrastructure.llm.factory import get_llm_provider
    from app.infrastructure.vectorstore.chroma_adapter import ChromaAdapter
    from app.infrastructure.websocket.connection_manager import ConnectionManager
    from app.rag.embeddings import EmbeddingService

    settings = get_settings()

    # ── Startup ──────────────────────────────────────────────────
    _configure_structlog(settings.app_log_level)

    # C6: Reject default JWT secret in non-test environments
    if settings.app_env not in ("test", "testing") and settings.jwt_secret_key == _DEFAULT_JWT_SECRET:
        logger.critical(
            "jwt_secret_not_configured",
            hint="Set JWT_SECRET_KEY env var to a strong random value",
        )
        msg = "JWT_SECRET_KEY must be changed from the default value in non-test environments"
        raise RuntimeError(msg)

    logger.info(
        "starting_application",
        app_name=settings.app_name,
        environment=settings.app_env,
        debug=settings.app_debug,
    )

    # Initialize Redis cache adapter (m3: decode_responses=True to avoid manual decoding)
    import redis.asyncio as aioredis

    from app.infrastructure.cache.redis_adapter import RedisAdapter

    try:
        redis_client = aioredis.from_url(
            settings.computed_redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
        )
        await redis_client.ping()  # type: ignore[misc]
        app.state.cache = RedisAdapter(redis_client)
        masked_url = _REDIS_PASSWORD_RE.sub(r"\1:***@", settings.computed_redis_url)
        logger.info("redis_connected", url=masked_url)
    except Exception:
        logger.warning("redis_connection_failed", exc_info=True)
        # Create a no-op cache that always returns None
        app.state.cache = None

    # Initialize LLM provider, vector store, embedding service, and ChatService
    llm_provider = get_llm_provider(settings)
    vector_store = ChromaAdapter(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection_prefix=settings.chroma_collection_prefix,
    )
    embedding_service = EmbeddingService(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
        cf_client_id=settings.cf_ollama_id,
        cf_client_secret=settings.cf_ollama_secret,
    )

    app.state.chat_service = ChatService(
        llm_provider=llm_provider,
        vector_store=vector_store,
        embedding_service=embedding_service,
    )
    app.state.llm_provider = llm_provider  # kept for cleanup
    app.state.embedding_service = embedding_service  # for ingestion worker
    app.state.vector_store = vector_store  # for ingestion worker
    logger.info("chat_service_initialized")

    # Initialize WebSocket connection manager
    app.state.ws_manager = ConnectionManager()
    logger.info("ws_manager_initialized")

    yield

    # ── Shutdown (m6: log shutdown first, then clean up resources) ──
    logger.info("shutting_down_application")

    # Close LLM adapter (httpx client)
    if hasattr(app.state, "llm_provider") and hasattr(app.state.llm_provider, "close"):
        await app.state.llm_provider.close()
        logger.info("llm_provider_closed")

    # Close embedding service (httpx client)
    if hasattr(app.state, "embedding_service") and hasattr(app.state.embedding_service, "close"):
        await app.state.embedding_service.close()
        logger.info("embedding_service_closed")

    if getattr(app.state, "cache", None) is not None:
        await app.state.cache.close()
        logger.info("redis_disconnected")
