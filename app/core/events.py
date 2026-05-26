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

# Default encryption secret — used to detect unchanged secrets at startup
_DEFAULT_SECRET_KEY = "change-me-to-a-random-secret-key"  # noqa: S105

# Regex to mask passwords in Redis URLs (redis://:password@host → redis://:***@host)
_REDIS_PASSWORD_RE = re.compile(r"(rediss?://):([^@]+)@")


import logging.config
import os

def _configure_structlog(log_level: str) -> None:
    """Configure structlog for JSON output with request-ID correlation."""
    os.makedirs("logs", exist_ok=True)
    
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.processors.JSONRenderer(),
            },
            "console": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.dev.ConsoleRenderer(colors=True),
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "console",
            },
            "file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": "logs/api.log",
                "when": "midnight",
                "backupCount": 30,
                "formatter": "json",
            },
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": log_level.upper(),
            },
            "uvicorn": {
                "handlers": ["console", "file"],
                "level": log_level.upper(),
                "propagate": False,
            },
        }
    })

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
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
    from app.workers.ingestion_queue import IngestionQueue

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

    # Phase 3: Reject default encryption secret in non-test environments
    if settings.app_env not in ("test", "testing") and settings.secret_key == _DEFAULT_SECRET_KEY:
        logger.critical(
            "secret_key_not_configured",
            hint="Set SECRET_KEY env var to a Fernet-compatible key",
        )
        msg = "SECRET_KEY must be changed from the default value in non-test environments"
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

    # Initialize ingestion queue with bounded concurrency
    app.state.ingestion_queue = IngestionQueue(
        max_concurrent=settings.ingestion_max_concurrent,
    )

    # ── Voice Pipeline (optional) ────────────────────────────────
    await _init_voice_providers(app, settings)

    # ── Superadmin Auto-Bootstrap ────────────────────────────────
    await _bootstrap_superadmin(settings)

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

    # Close shared webhook HTTP client
    try:
        from app.rag.tools.webhook import close_shared_http_client
        await close_shared_http_client()
        logger.info("webhook_client_closed")
    except Exception:
        logger.warning("webhook_client_close_failed", exc_info=True)


async def _init_voice_providers(app: FastAPI, settings: object) -> None:
    """Try to initialize local STT/TTS providers.

    Graceful: if voice dependencies (faster-whisper, piper-tts) are not
    installed, the app starts normally with voice disabled. The voice API
    endpoints use ``getattr(app.state, "stt_provider", None)`` so they
    handle the missing attribute cleanly.
    """
    from app.infrastructure.voice.pipeline_factory import VoiceSessionManager

    app.state.stt_provider = None
    app.state.tts_provider = None
    app.state.voice_session_manager = VoiceSessionManager(
        default_max_sessions=getattr(settings, "voice_max_sessions_per_tenant", 3),
    )

    # ── STT (Whisper) ────────────────────────────────────────────
    try:
        from app.infrastructure.stt.factory import get_stt_provider

        stt = get_stt_provider(
            "whisper",
            model_size=getattr(settings, "voice_stt_model", "base"),
            max_audio_bytes=getattr(settings, "voice_max_audio_bytes", 10 * 1024 * 1024),
        )
        await stt.warm_up()
        app.state.stt_provider = stt
        logger.info("voice_stt_initialized", provider="whisper")
    except Exception:
        logger.info("voice_stt_unavailable", reason="faster-whisper not installed or model load failed")

    # ── TTS (Piper) ──────────────────────────────────────────────
    try:
        from app.infrastructure.tts.factory import get_tts_provider

        tts = get_tts_provider(
            "piper",
            voice=getattr(settings, "voice_tts_voice", "en_US-lessac-medium"),
        )
        await tts.warm_up()
        app.state.tts_provider = tts
        logger.info("voice_tts_initialized", provider="piper")
    except Exception:
        logger.info("voice_tts_unavailable", reason="piper-tts not installed or model load failed")

    # Summary
    stt_ok = app.state.stt_provider is not None
    tts_ok = app.state.tts_provider is not None
    if stt_ok and tts_ok:
        logger.info("voice_pipeline_ready")
    elif stt_ok or tts_ok:
        logger.warning("voice_pipeline_partial", stt=stt_ok, tts=tts_ok)
    else:
        logger.info("voice_pipeline_disabled", reason="no local providers available")


# ── Well-known management tenant ────────────────────────────────
_MANAGEMENT_TENANT_ID = "00000000-0000-0000-0000-000000000001"
_MANAGEMENT_TENANT_NAME = "Platform Management"
_MANAGEMENT_TENANT_SLUG = "management"


async def _bootstrap_superadmin(settings: object) -> None:
    """Auto-create management tenant and superadmin user on startup.

    Idempotent: skips creation if the tenant/user already exists.
    Only runs when both SUPERADMIN_EMAIL and SUPERADMIN_PASSWORD are
    set in the environment. This eliminates the need for the manual
    ``scripts/create_superadmin.py`` step.
    """
    email = getattr(settings, "superadmin_email", "")
    password = getattr(settings, "superadmin_password", "")

    if not email or not password:
        logger.debug("superadmin_bootstrap_skipped", reason="SUPERADMIN_EMAIL or SUPERADMIN_PASSWORD not set")
        return

    from app.core.security import hash_password
    from app.domain.models.enums import TenantStatus, UserRole
    from app.domain.models.tenant import TenantCreate
    from app.domain.models.user import UserCreate
    from app.infrastructure.database.connection import AsyncSessionLocal
    from app.infrastructure.database.repositories.tenant_repo import SQLTenantRepository
    from app.infrastructure.database.repositories.user_repo import SQLUserRepository

    try:
        async with AsyncSessionLocal() as session:
            tenant_repo = SQLTenantRepository(session)
            user_repo = SQLUserRepository(session)

            # 1. Ensure management tenant exists
            existing_tenant = await tenant_repo.get_by_id(_MANAGEMENT_TENANT_ID)
            if existing_tenant is None:
                tenant_data = TenantCreate(
                    name=_MANAGEMENT_TENANT_NAME,
                    slug=_MANAGEMENT_TENANT_SLUG,
                    config_json={},
                    status=TenantStatus.ACTIVE,
                )
                # Use the well-known ID by inserting directly
                from app.infrastructure.database.models import TenantModel

                tenant_model = TenantModel(
                    id=_MANAGEMENT_TENANT_ID,
                    name=tenant_data.name,
                    slug=tenant_data.slug,
                    config_json=tenant_data.config_json,
                    status=TenantStatus.ACTIVE,
                )
                session.add(tenant_model)
                await session.flush()
                logger.info("management_tenant_created", tenant_id=_MANAGEMENT_TENANT_ID)
            else:
                logger.debug("management_tenant_exists", tenant_id=_MANAGEMENT_TENANT_ID)

            # 2. Ensure superadmin user exists
            existing_user = await user_repo.get_by_email(email, _MANAGEMENT_TENANT_ID)
            if existing_user is None:
                hashed = hash_password(password)
                user_create = UserCreate(email=email, role=UserRole.SUPERADMIN)
                user = await user_repo.create(_MANAGEMENT_TENANT_ID, user_create, password_hash=hashed)
                logger.info("superadmin_bootstrapped", email=email, user_id=user.id)
            else:
                logger.debug("superadmin_exists", email=email)

            await session.commit()

    except Exception:
        logger.warning("superadmin_bootstrap_failed", exc_info=True)
