"""Widget API schemas — request/response DTOs for the embeddable SDK."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class WidgetSessionRequest(BaseModel):
    """Request body for POST /api/v1/widget/session."""

    tenant_slug: str = Field(
        ...,
        min_length=2,
        max_length=63,
        description="Tenant slug (e.g., 'medforge')",
    )
    embed_key: str = Field(
        ...,
        min_length=1,
        description="Embed key from tenant config (e.g., 'pk_live_...')",
    )
    visitor_id: str = Field(
        default="",
        max_length=255,
        description="Optional visitor identifier for session continuity",
    )


class WidgetSessionResponse(BaseModel):
    """Response body for POST /api/v1/widget/session."""

    session_token: str = Field(..., description="ws_-prefixed session token for WebSocket")
    tenant_id: str = Field(..., description="Tenant UUID")
    tenant_slug: str = Field(..., description="Tenant slug")
    expires_in: int = Field(..., description="Token TTL in seconds")


class WidgetUIConfigResponse(BaseModel):
    """Response body for GET /api/v1/widget/ui-config/{slug}.

    Returns only public-facing UI config — never exposes tools,
    secrets, agent_prompt, or other internal tenant configuration.
    """

    brand_name: str = Field(default="Support", description="Brand name for the widget header")
    logo_url: str = Field(default="", description="Brand logo URL")
    welcome_message: str = Field(
        default="Hi! How can I help you today?",
        description="Welcome message shown when chat opens",
    )
    placeholder_text: str = Field(
        default="Type your message...",
        description="Input placeholder text",
    )
    theme: dict[str, Any] = Field(default_factory=dict, description="Theme configuration")
    widget: dict[str, Any] = Field(default_factory=dict, description="Widget behavior settings")
