"""Model management API router — admin-only model listing and selection.

Provides endpoints for admins to:
- List all available LLM models from configured providers
- Get/set the active chat model

All endpoints enforce admin-only access via ``require_role(UserRole.ADMIN)``.
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

if TYPE_CHECKING:
    from app.domain.models.user import User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["models"])


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    request: Request,
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> ModelListResponse:
    """List all available chat models grouped by provider.

    Queries each registered LLM provider for its available models
    and returns them grouped. Embedding-only models are filtered out.

    Args:
        request: FastAPI request (for accessing app state).
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

    # Current active model
    active = ActiveModel(
        provider="ollama",
        model_id=getattr(llm_provider, "default_model", ""),
    )

    logger.debug(
        "models_listed",
        user_id=user.id,
        model_count=len(ollama_models),
        active_model=active.model_id,
    )

    return ModelListResponse(providers=providers, active_model=active)


@router.put("/models/active", response_model=SetActiveModelResponse)
async def set_active_model(
    body: SetActiveModelRequest,
    request: Request,
    user: User = Depends(require_role(UserRole.ADMIN)),
) -> SetActiveModelResponse:
    """Set the active chat model.

    Updates the LLM provider's default model at runtime. No server
    restart required. The change takes effect for all subsequent
    chat requests.

    Validates that the requested model actually exists in the
    provider's available model list before activating.

    Args:
        body: Provider and model ID to activate.
        request: FastAPI request (for accessing app state).
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

    # Update the default model at runtime
    old_model = getattr(llm_provider, "default_model", "")
    llm_provider.default_model = body.model_id

    logger.info(
        "active_model_changed",
        old_model=old_model,
        new_model=body.model_id,
        changed_by=user.id,
        provider=body.provider,
    )

    return SetActiveModelResponse(
        provider=body.provider,
        model_id=body.model_id,
        status="active",
    )
