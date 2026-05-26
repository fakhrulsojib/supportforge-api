"""ToolExecutor — safety guardrails for tool execution.

Handles timeouts, response size limits, error isolation, and audit logging.
Graceful degradation: on failure, returns error as ToolResult so the LLM
can generate a user-friendly message.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from app.rag.tools.base import ToolResult

logger = structlog.get_logger(__name__)


class ToolExecutor:
    """Execute tools with safety guardrails."""

    DEFAULT_MAX_ROUNDS: int = 3
    MAX_RESPONSE_BYTES: int = 50_000

    def __init__(self, max_rounds: int = DEFAULT_MAX_ROUNDS) -> None:
        self.max_rounds = max_rounds

    async def execute(
        self,
        tool: Any,  # WebhookTool — avoid circular import
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Execute a tool call with safety guardrails.

        Handles:
        - Built-in escalate tool (returns immediately)
        - Timeouts (per-tool configurable)
        - Response size limits (truncates huge responses)
        - Error isolation (exceptions → ToolResult with error)
        """
        # Handle built-in escalate tool
        if (
            hasattr(tool, "definition")
            and tool.definition.is_builtin
            and tool.definition.name == "escalate"
        ):
            return ToolResult(
                success=True,
                data={"escalated": True, "reason": arguments.get("reason", "")},
            )

        try:
            timeout = getattr(getattr(tool, "config", None), "timeout", 15.0)
            result = await asyncio.wait_for(
                tool.execute(arguments), timeout=timeout
            )

            # Truncate oversized responses
            try:
                response_size = len(json.dumps(result.data))
            except (TypeError, ValueError):
                response_size = 0

            if response_size > self.MAX_RESPONSE_BYTES:
                logger.warning(
                    "tool_response_truncated",
                    tool_name=getattr(tool, "config", None) and tool.config.name,
                    size=response_size,
                )
                result = ToolResult(
                    success=True,
                    data={
                        "_truncated": True,
                        "partial": json.dumps(result.data)[:5000],
                    },
                    execution_time_ms=result.execution_time_ms,
                )

            logger.info(
                "tool_executed",
                tool_name=getattr(tool, "config", None) and tool.config.name,
                success=result.success,
                time_ms=result.execution_time_ms,
            )
            return result

        except asyncio.TimeoutError:
            tool_name = getattr(tool, "config", None) and tool.config.name
            logger.warning("tool_timeout", tool_name=tool_name)
            return ToolResult(
                success=False,
                data={},
                error="Tenant API timed out — please try again",
            )
        except Exception as exc:
            tool_name = getattr(tool, "config", None) and tool.config.name
            logger.error(
                "tool_execution_failed",
                tool_name=tool_name,
                error=str(exc),
                exc_info=True,
            )
            return ToolResult(
                success=False,
                data={},
                error="Tool execution failed — please try again",
            )
