"""Model management API router — admin-only model listing and selection.

Provides endpoints for admins to:
- List all available LLM models from configured providers
- Get/set the active chat model (persisted per tenant in config_json)

All endpoints enforce admin-only access via ``require_role(UserRole.ADMIN)``.
Model selection is tenant-scoped and persisted to the database.
Designed to be extensible for future providers (OpenAI, Gemini, etc.).
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
from app.core.dependencies import require_role
from app.core.exceptions import SupportForgeError
from app.domain.models.enums import UserRole
from app.infrastructure.database.connection import get_async_session
from app.infrastructure.database.repositories.tenant_repo import SQLTenantRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.domain.models.user import User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["models"])


def _get_tenant_chat_model(config_json: dict, fallback: str) -> str:
    """Extract chat_model from tenant config_json, falling back to server default."""
    return str(config_json.get("chat_model", "")) or fallback


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> ModelListResponse:
    """List all available chat models grouped by provider.

    Queries each registered LLM provider for its available models
    and returns them grouped. The active model is read from the
    authenticated user's tenant config (persisted per tenant).

    Args:
        request: FastAPI request (for accessing app state).
        session: Database session.
        user: Authenticated admin user.

    Returns:
        Available models grouped by provider, plus the active model.
    """
    llm_provider = request.app.state.llm_provider

    # Query Ollama for available models
    raw_models = await llm_provider.list_models()

    ollama_models = [
        ModelInfo(
            id=str(m["id"]),
            name=str(m["name"]),
            size_gb=float(m.get("size_gb", 0)),
        )
        for m in raw_models
    ]

    providers = [
        ProviderInfo(
            id="ollama",
            name="Ollama (Self-hosted)",
            models=ollama_models,
        ),
    ]

    # Read tenant's active model from config_json (persisted)
    tenant_repo = SQLTenantRepository(session)
    tenant = await tenant_repo.get_by_id(user.tenant_id)
    server_default = getattr(llm_provider, "default_model", "")
    active_model_id = _get_tenant_chat_model(
        tenant.config_json if tenant else {},
        server_default,
    )

    active = ActiveModel(provider="ollama", model_id=active_model_id)

    logger.debug(
        "models_listed",
        user_id=user.id,
        tenant_id=user.tenant_id,
        model_count=len(ollama_models),
        active_model=active.model_id,
    )

    return ModelListResponse(providers=providers, active_model=active)


@router.put("/models/active", response_model=SetActiveModelResponse)
async def set_active_model(
    body: SetActiveModelRequest,
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> SetActiveModelResponse:
    """Set the active chat model for the current tenant.

    Persists the model selection to the tenant's ``config_json`` in the
    database. This is tenant-scoped — each tenant can have a different
    active model. The change takes effect for all subsequent chat
    requests within this tenant.

    Validates that the requested model actually exists in the
    provider's available model list before activating.

    Args:
        body: Provider and model ID to activate.
        request: FastAPI request (for accessing app state).
        session: Database session.
        user: Authenticated admin user.

    Returns:
        Confirmation of the activated model.

    Raises:
        SupportForgeError: If provider or model not found.
    """
    if body.provider != "ollama":
        raise SupportForgeError(
            message=f"Provider '{body.provider}' is not configured",
            status_code=404,
            error_code="PROVIDER_NOT_FOUND",
        )

    llm_provider = request.app.state.llm_provider

    # Validate model exists
    available = await llm_provider.list_models()
    model_ids = {str(m["id"]) for m in available}

    if body.model_id not in model_ids:
        raise SupportForgeError(
            message=f"Model '{body.model_id}' not found in provider '{body.provider}'",
            status_code=404,
            error_code="MODEL_NOT_FOUND",
        )

    # Persist to tenant config_json
    tenant_repo = SQLTenantRepository(session)
    tenant = await tenant_repo.get_by_id(user.tenant_id)
    if not tenant:
        raise SupportForgeError(
            message="Tenant not found",
            status_code=404,
            error_code="TENANT_NOT_FOUND",
        )

    # Merge into existing config_json (preserve other keys)
    updated_config = {**tenant.config_json, "chat_model": body.model_id}
    await tenant_repo.update(user.tenant_id, config_json=updated_config)
    await session.commit()

    logger.info(
        "active_model_changed",
        old_model=_get_tenant_chat_model(
            tenant.config_json,
            getattr(llm_provider, "default_model", ""),
        ),
        new_model=body.model_id,
        changed_by=user.id,
        tenant_id=user.tenant_id,
        provider=body.provider,
    )

    return SetActiveModelResponse(
        provider=body.provider,
        model_id=body.model_id,
        status="active",
    )
