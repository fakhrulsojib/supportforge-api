"""Model management API schemas — request/response DTOs."""

from __future__ import annotations

from typing import Literal

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
    models: list[ModelInfo] = Field(default_factory=list, description="Chat models")
    embedding_models: list[ModelInfo] = Field(
        default_factory=list, description="Embedding models",
    )


class ActiveModel(BaseModel):
    """The currently active models for a tenant."""

    provider: str = Field(..., description="Provider identifier")
    model_id: str = Field(..., description="Active chat model identifier")
    embedding_model_id: str = Field("", description="Active embedding model identifier")


class ModelListResponse(BaseModel):
    """Response for listing all available models."""

    providers: list[ProviderInfo] = Field(..., description="Available providers and their models")
    active_model: ActiveModel = Field(..., description="Currently active models for this tenant")


class SetActiveModelRequest(BaseModel):
    """Request to set an active model (chat or embedding)."""

    provider: str = Field(..., description="Provider identifier (e.g. 'ollama')")
    model_id: str = Field(..., description="Model identifier (e.g. 'gemma3:4b')")
    model_type: Literal["chat", "embedding"] = Field(
        "chat", description="Model type: 'chat' or 'embedding'",
    )


class SetActiveModelResponse(BaseModel):
    """Response after setting an active model."""

    provider: str = Field(..., description="Provider identifier")
    model_id: str = Field(..., description="Model identifier")
    model_type: Literal["chat", "embedding"] = Field("chat", description="Model type that was changed")
    status: str = Field("active", description="Activation status")
