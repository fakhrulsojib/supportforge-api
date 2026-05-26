"""Pydantic validators for tenant ``config_json``.

Validates tool definitions, prompt config, and other structured
sections of ``config_json`` on tenant update.  Unknown keys pass
through unvalidated for backward compatibility.
"""

from __future__ import annotations

import urllib.parse
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ToolConfigItem(BaseModel):
    """A single tool definition in ``config_json.tools[]``."""

    name: str
    description: str
    http_method: str = "GET"
    endpoint: str  # Relative or absolute URL
    parameters: dict[str, Any] = Field(default_factory=dict)
    requires_confirmation: bool = False
    timeout: float = 15.0
    response_mapping: dict[str, str] | None = None
    auth_header: str = "Authorization"

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint_scheme(cls, v: str) -> str:
        """Block non-HTTP(S) schemes at config time (SSRF prevention)."""
        if v and v.startswith(("http://", "https://", "/")):
            return v
        if v and "://" in v:
            scheme = urllib.parse.urlparse(v).scheme
            msg = f"Only HTTP(S) schemes allowed for endpoints, got '{scheme}'"
            raise ValueError(msg)
        # Relative paths (no scheme) are OK — will be prepended with base_url
        return v

    @field_validator("http_method")
    @classmethod
    def validate_http_method(cls, v: str) -> str:
        allowed = {"GET", "POST", "PUT", "PATCH", "DELETE"}
        upper = v.upper()
        if upper not in allowed:
            msg = f"http_method must be one of {allowed}, got '{v}'"
            raise ValueError(msg)
        return upper

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: float) -> float:
        if v <= 0 or v > 120:
            msg = "timeout must be between 0 and 120 seconds"
            raise ValueError(msg)
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or len(v) > 100:
            msg = "name must be 1-100 characters"
            raise ValueError(msg)
        return v


class TenantConfigValidator(BaseModel):
    """Optional validation — only validates keys that are present.

    Unknown keys pass through unvalidated for backward compatibility.
    """

    tools: list[ToolConfigItem] | None = None
    tools_enabled: bool | None = None
    tools_base_url: str | None = None
    max_tool_rounds: int | None = None
    agent_prompt: dict[str, Any] | None = None
    ui_config: dict[str, Any] | None = None
    event_hooks: dict[str, Any] | None = None

    model_config = {"extra": "allow"}

    @field_validator("max_tool_rounds")
    @classmethod
    def validate_max_tool_rounds(cls, v: int | None) -> int | None:
        if v is not None and (v < 1 or v > 10):
            msg = "max_tool_rounds must be between 1 and 10"
            raise ValueError(msg)
        return v

    @field_validator("tools_base_url")
    @classmethod
    def validate_tools_base_url(cls, v: str | None) -> str | None:
        """Block non-HTTP(S) schemes on the base URL."""
        if v and v.startswith(("http://", "https://")):
            return v
        if v and "://" in v:
            scheme = urllib.parse.urlparse(v).scheme
            msg = f"Only HTTP(S) schemes allowed for tools_base_url, got '{scheme}'"
            raise ValueError(msg)
        return v

    @field_validator("tools")
    @classmethod
    def validate_tools_unique_names(
        cls, v: list[ToolConfigItem] | None
    ) -> list[ToolConfigItem] | None:
        if v is None:
            return v
        if len(v) > 20:
            msg = "Maximum 20 tools per tenant allowed"
            raise ValueError(msg)
        names = [t.name for t in v]
        if len(names) != len(set(names)):
            msg = "Tool names must be unique"
            raise ValueError(msg)
        return v


def validate_config_json(config: dict[str, Any]) -> dict[str, Any]:
    """Validate ``config_json`` on tenant update.

    Returns the validated dict.  Raises ``ValidationError`` on bad config,
    or ``TypeError`` if ``config`` is not a dict.
    """
    if not isinstance(config, dict):
        msg = f"config_json must be a dict, got {type(config).__name__}"
        raise TypeError(msg)
    validated = TenantConfigValidator(**config)
    # Preserve caller-provided keys + validated results; don't add null fields
    # that weren't in the original config.
    return validated.model_dump(exclude_unset=True)
