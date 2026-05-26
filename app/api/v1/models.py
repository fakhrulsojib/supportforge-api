"""Model management API router — admin-only model listing and selection.

Provides endpoints for admins to:
- List all available LLM and embedding models from configured providers
- Get/set the active chat model (persisted per tenant in config_json)
- Get/set the active embedding model (persisted per tenant in config_json)

All endpoints enforce admin-only access via ``require_role(UserRole.ADMIN)``.
Model selection is tenant-scoped and persisted to the database.

Supports multiple providers:
- **Ollama** — self-hosted, models listed from /api/tags
- **Gemini** — Google cloud, static model catalog, requires per-tenant API key
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends, Request

from app.api.schemas.models import (
    ActiveModel,
    ModelInfo,
    ModelListResponse,
    ProviderInfo,
    SetActiveModelRequest,
    SetActiveModelResponse,
)
from app.config import get_settings
from app.core.crypto import decrypt_value, encrypt_value, mask_api_key
from app.core.dependencies import require_role
from app.core.exceptions import SupportForgeError
from app.core.tenant_config import (
    CONFIG_CHAT_MODEL,
    CONFIG_CHAT_PROVIDER,
    CONFIG_EMBEDDING_MODEL,
    CONFIG_EMBEDDING_PROVIDER,
    CONFIG_GEMINI_API_KEY,
    CONFIG_GEMINI_EMBEDDING_API_KEY,
)
from app.domain.models.enums import UserRole
from app.infrastructure.database.connection import get_async_session
from app.infrastructure.database.repositories.tenant_repo import SQLTenantRepository
from app.infrastructure.llm.gemini_adapter import _GEMINI_CHAT_MODELS
from app.infrastructure.llm.gemini_embedding_adapter import _GEMINI_EMBEDDING_MODELS

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.domain.models.user import User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["models"])

# Known providers and their identifiers
_SUPPORTED_PROVIDERS = {"ollama", "gemini"}


def _get_tenant_model(config_json: dict | None, key: str, fallback: str) -> str:
    """Extract a model ID from tenant config_json, falling back to server default."""
    if not config_json:
        return fallback
    return str(config_json.get(key, "")) or fallback


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> ModelListResponse:
    """List all available chat and embedding models grouped by provider.

    Queries each registered LLM provider for its available models
    and returns them grouped. The active models are read from the
    authenticated user's tenant config (persisted per tenant).

    Args:
        request: FastAPI request (for accessing app state).
        session: Database session.
        user: Authenticated admin user.

    Returns:
        Available models grouped by provider, plus the active models.
    """
    logger.debug("api_models_list", tenant_id=user.tenant_id, user_id=user.id)
    llm_provider = request.app.state.llm_provider

    # ── Ollama provider ─────────────────────────────────────────
    raw_chat_models = await llm_provider.list_models()
    raw_embedding_models = await llm_provider.list_embedding_models()

    ollama_chat = [
        ModelInfo(
            id=str(m["id"]),
            name=str(m["name"]),
            size_gb=float(m.get("size_gb", 0)),
        )
        for m in raw_chat_models
    ]

    ollama_embed = [
        ModelInfo(
            id=str(m["id"]),
            name=str(m["name"]),
            size_gb=float(m.get("size_gb", 0)),
        )
        for m in raw_embedding_models
    ]

    # ── Gemini provider (static catalog — no adapter needed) ────
    gemini_chat = [
        ModelInfo(
            id=str(m["id"]),
            name=str(m["name"]),
            size_gb=float(m.get("size_gb", 0)),
        )
        for m in _GEMINI_CHAT_MODELS
    ]

    gemini_embed = [
        ModelInfo(
            id=str(m["id"]),
            name=str(m["name"]),
            size_gb=float(m.get("size_gb", 0)),
        )
        for m in _GEMINI_EMBEDDING_MODELS
    ]

    providers = [
        ProviderInfo(
            id="ollama",
            name="Ollama (Self-hosted)",
            models=ollama_chat,
            embedding_models=ollama_embed,
        ),
        ProviderInfo(
            id="gemini",
            name="Google Gemini",
            models=gemini_chat,
            embedding_models=gemini_embed,
        ),
    ]

    # Read tenant's active models from config_json (persisted)
    tenant_repo = SQLTenantRepository(session)
    tenant = await tenant_repo.get_by_id(user.tenant_id)
    config = (tenant.config_json if tenant else None) or {}

    server_chat_default = getattr(llm_provider, "default_model", "")
    embedding_service = getattr(request.app.state, "embedding_service", None)
    server_embed_default = getattr(embedding_service, "model", "") if embedding_service else ""

    # Determine active provider
    active_provider = str(config.get(CONFIG_CHAT_PROVIDER, "ollama")) if config else "ollama"
    if active_provider not in _SUPPORTED_PROVIDERS:
        active_provider = "ollama"

    # Check for Gemini chat API key (masked preview only)
    settings = get_settings()
    has_key = False
    key_preview = ""
    raw_encrypted_key = config.get(CONFIG_GEMINI_API_KEY) if config else None
    if isinstance(raw_encrypted_key, str) and raw_encrypted_key:
        has_key = True
        try:
            decrypted = decrypt_value(raw_encrypted_key, settings.secret_key)
            key_preview = mask_api_key(decrypted)
        except Exception:
            key_preview = "****"

    # Check for Gemini embedding API key (separate key)
    has_embed_key = False
    embed_key_preview = ""
    raw_embed_encrypted = config.get(CONFIG_GEMINI_EMBEDDING_API_KEY) if config else None
    if isinstance(raw_embed_encrypted, str) and raw_embed_encrypted:
        has_embed_key = True
        try:
            decrypted = decrypt_value(raw_embed_encrypted, settings.secret_key)
            embed_key_preview = mask_api_key(decrypted)
        except Exception:
            embed_key_preview = "****"

    # Determine active embedding provider
    active_embed_provider = str(config.get(CONFIG_EMBEDDING_PROVIDER, "ollama")) if config else "ollama"
    if active_embed_provider not in _SUPPORTED_PROVIDERS:
        active_embed_provider = "ollama"

    active = ActiveModel(
        provider=active_provider,
        model_id=_get_tenant_model(config, CONFIG_CHAT_MODEL, server_chat_default),
        embedding_model_id=_get_tenant_model(config, CONFIG_EMBEDDING_MODEL, server_embed_default),
        has_api_key=has_key,
        api_key_preview=key_preview,
        embedding_provider=active_embed_provider,
        has_embedding_api_key=has_embed_key,
        embedding_api_key_preview=embed_key_preview,
    )

    logger.debug(
        "models_listed",
        user_id=user.id,
        tenant_id=user.tenant_id,
        ollama_models=len(ollama_chat),
        gemini_models=len(gemini_chat),
        active_provider=active_provider,
        active_chat=active.model_id,
    )

    return ModelListResponse(providers=providers, active_model=active)


@router.put("/models/active", response_model=SetActiveModelResponse)
async def set_active_model(
    body: SetActiveModelRequest,
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> SetActiveModelResponse:
    """Set the active chat or embedding model for the current tenant.

    Persists the model selection to the tenant's ``config_json`` in the
    database. This is tenant-scoped — each tenant can have a different
    active model. The change takes effect for all subsequent requests
    within this tenant.

    For Gemini models, an ``api_key`` must be provided on first setup.
    The key is encrypted before storage.

    Args:
        body: Provider, model ID, model type, and optional API key.
        request: FastAPI request (for accessing app state).
        session: Database session.
        user: Authenticated admin user.

    Returns:
        Confirmation of the activated model.

    Raises:
        SupportForgeError: If provider, model, or type not found.
    """
    logger.debug("api_models_set_active", tenant_id=user.tenant_id, provider=body.provider, model_id=body.model_id, model_type=body.model_type)
    if body.provider not in _SUPPORTED_PROVIDERS:
        raise SupportForgeError(
            message=f"Provider '{body.provider}' is not configured",
            status_code=404,
            error_code="PROVIDER_NOT_FOUND",
        )

    # ── Validate model exists ───────────────────────────────────
    if body.provider == "ollama":
        llm_provider = request.app.state.llm_provider
        if body.model_type == "chat":
            available = await llm_provider.list_models()
            config_key = CONFIG_CHAT_MODEL
        else:
            available = await llm_provider.list_embedding_models()
            config_key = CONFIG_EMBEDDING_MODEL
        model_ids = {str(m["id"]) for m in available}
    elif body.provider == "gemini":
        if body.model_type == "chat":
            available = _GEMINI_CHAT_MODELS
            config_key = CONFIG_CHAT_MODEL
        else:
            available = list(_GEMINI_EMBEDDING_MODELS)
            config_key = CONFIG_EMBEDDING_MODEL
        model_ids = {str(m["id"]) for m in available}
    else:
        model_ids = set()
        config_key = CONFIG_CHAT_MODEL

    if body.model_id not in model_ids:
        raise SupportForgeError(
            message=f"Model '{body.model_id}' not found in provider '{body.provider}' ({body.model_type} models)",
            status_code=404,
            error_code="MODEL_NOT_FOUND",
        )

    # ── Gemini-specific: validate API key exists ─────────────────
    settings = get_settings()

    # ── Persist to tenant config_json ───────────────────────────
    tenant_repo = SQLTenantRepository(session)
    tenant = await tenant_repo.get_by_id(user.tenant_id)
    if not tenant:
        raise SupportForgeError(
            message="Tenant not found",
            status_code=404,
            error_code="TENANT_NOT_FOUND",
        )

    if body.provider == "gemini" and body.model_type == "chat":
        existing_key = (
            tenant.config_json.get(CONFIG_GEMINI_API_KEY)
            if tenant.config_json
            else None
        )
        if not body.api_key and not existing_key:
            raise SupportForgeError(
                message="Gemini API key is required when setting a Gemini model",
                status_code=400,
                error_code="API_KEY_REQUIRED",
            )

    if body.provider == "gemini" and body.model_type == "embedding":
        existing_embed_key = (
            tenant.config_json.get(CONFIG_GEMINI_EMBEDDING_API_KEY)
            if tenant.config_json
            else None
        )
        if not body.api_key and not existing_embed_key:
            raise SupportForgeError(
                message="Gemini embedding API key is required when setting a Gemini embedding model",
                status_code=400,
                error_code="API_KEY_REQUIRED",
            )

    old_config = tenant.config_json or {}
    old_value = old_config.get(config_key, "")
    updated_config = {**old_config, config_key: body.model_id}

    # Set provider for chat model changes
    if body.model_type == "chat":
        updated_config[CONFIG_CHAT_PROVIDER] = body.provider

        # Encrypt and store API key if provided
        if body.api_key and body.provider == "gemini":
            encrypted_key = encrypt_value(body.api_key, settings.secret_key)
            updated_config[CONFIG_GEMINI_API_KEY] = encrypted_key

        # Clear Gemini key when switching back to Ollama
        if body.provider == "ollama" and CONFIG_GEMINI_API_KEY in updated_config:
            del updated_config[CONFIG_GEMINI_API_KEY]

    # Set provider for embedding model changes
    if body.model_type == "embedding":
        updated_config[CONFIG_EMBEDDING_PROVIDER] = body.provider

        # Encrypt and store embedding API key if provided
        if body.api_key and body.provider == "gemini":
            encrypted_key = encrypt_value(body.api_key, settings.secret_key)
            updated_config[CONFIG_GEMINI_EMBEDDING_API_KEY] = encrypted_key

        # Clear Gemini embedding key when switching back to Ollama
        if body.provider == "ollama" and CONFIG_GEMINI_EMBEDDING_API_KEY in updated_config:
            del updated_config[CONFIG_GEMINI_EMBEDDING_API_KEY]

    await tenant_repo.update(user.tenant_id, config_json=updated_config)
    await session.commit()

    logger.info(
        "active_model_changed",
        model_type=body.model_type,
        old_model=old_value,
        new_model=body.model_id,
        changed_by=user.id,
        tenant_id=user.tenant_id,
        provider=body.provider,
    )

    return SetActiveModelResponse(
        provider=body.provider,
        model_id=body.model_id,
        model_type=body.model_type,
        status="active",
    )
