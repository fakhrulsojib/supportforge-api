"""Model management API schemas — request/response DTOs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """A single available model."""

    id: str = Field(..., description="Model identifier for API calls")
    name: str = Field(..., description="Display name")
    size_gb: float = Field(0, description="Approximate model size in GB")


class ProviderInfo(BaseModel):
    """A model provider with its available models."""

    id: str = Field(..., description="Provider identifier (e.g. 'ollama')")
    name: str = Field(..., description="Display name (e.g. 'Ollama (Self-hosted)')")
    models: list[ModelInfo] = Field(default_factory=list, description="Available models")


class ActiveModel(BaseModel):
    """The currently active chat model."""

    provider: str = Field(..., description="Provider identifier")
    model_id: str = Field(..., description="Model identifier")


class ModelListResponse(BaseModel):
    """Response for listing all available models."""

    providers: list[ProviderInfo] = Field(..., description="Available providers and their models")
    active_model: ActiveModel = Field(..., description="Currently active chat model")


class SetActiveModelRequest(BaseModel):
    """Request to set the active chat model."""

    provider: str = Field(..., description="Provider identifier (e.g. 'ollama')")
    model_id: str = Field(..., description="Model identifier (e.g. 'gemma3:4b')")


class SetActiveModelResponse(BaseModel):
    """Response after setting the active model."""

    provider: str = Field(..., description="Provider identifier")
    model_id: str = Field(..., description="Model identifier")
    status: str = Field("active", description="Activation status")
