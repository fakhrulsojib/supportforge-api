"""Resolve tenant config_json into executable WebhookTool instances.

Converts the JSON tool definitions in ``config_json.tools[]`` into
``WebhookTool`` objects ready for execution.
"""
from __future__ import annotations

from typing import Any

import structlog

from app.rag.tools.base import ESCALATE_TOOL_DEFINITION, ToolDefinition, ToolResult
from app.rag.tools.webhook import WebhookTool, WebhookToolConfig

logger = structlog.get_logger(__name__)


class BuiltinEscalateTool:
    """Thin wrapper so the escalate tool has the same interface as WebhookTool."""

    @property
    def definition(self) -> ToolDefinition:
        return ESCALATE_TOOL_DEFINITION

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=True,
            data={"escalated": True, "reason": arguments.get("reason", "")},
        )


def resolve_tenant_tools(
    config_json: dict[str, Any] | None,
    *,
    secrets: dict[str, str] | None = None,
) -> list[Any]:
    """Convert tenant config_json into executable tool instances.

    Returns a list that always includes the built-in ``escalate`` tool.
    If ``tools_enabled`` is false or missing, returns only ``[escalate]``.

    Args:
        config_json: Tenant configuration dict.
        secrets: Decrypted secrets from tenant_secrets table,
            keyed by name (e.g., ``{"tools_auth.default": "Bearer xxx"}``).

    Returns:
        List of tool instances (WebhookTool and/or BuiltinEscalateTool).
    """
    escalate_tool = BuiltinEscalateTool()

    if not config_json or not config_json.get("tools_enabled"):
        return [escalate_tool]

    secrets = secrets or {}
    default_auth = secrets.get("tools_auth.default", "")
    base_url = config_json.get("tools_base_url", "")
    tools: list[Any] = []

    for t in config_json.get("tools", []):
        try:
            name = t["name"]
            description = t["description"]
        except (KeyError, TypeError):
            logger.warning("tool_config_invalid", tool=t)
            continue

        endpoint = t.get("endpoint", "")
        full_url = (
            endpoint if endpoint.startswith("http") else f"{base_url}{endpoint}"
        )

        # Per-tool auth from secrets, with fallback to tenant-level default
        tool_auth = secrets.get(f"tools_auth.{name}", default_auth)

        tools.append(
            WebhookTool(
                config=WebhookToolConfig(
                    name=name,
                    description=description,
                    http_method=t.get("http_method", "GET"),
                    endpoint_url=full_url,
                    parameters_schema=t.get("parameters", {}),
                    requires_confirmation=t.get("requires_confirmation", False),
                    timeout=t.get("timeout", 15.0),
                    response_mapping=t.get("response_mapping"),
                ),
                auth_header=t.get("auth_header", "Authorization"),
                auth_value=tool_auth,
            )
        )

    # Escalate tool always last — so LLM sees it but prefers tenant tools
    tools.append(escalate_tool)
    return tools
