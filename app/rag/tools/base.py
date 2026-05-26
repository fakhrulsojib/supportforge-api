"""Base types for the pluggable tool system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    """Describes a tool that the LLM can call."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    is_builtin: bool = False

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    """Result of executing a tool."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    execution_time_ms: float = 0


# Built-in escalate tool — always injected for every tenant.
ESCALATE_TOOL_DEFINITION = ToolDefinition(
    name="escalate",
    description=(
        "Escalate the conversation to a human support agent. "
        "Call this when the customer explicitly asks for a human, "
        "describes a situation requiring human judgment, or when you "
        "cannot resolve their issue from the available information."
    ),
    parameters={
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why escalation is needed",
            }
        },
        "required": ["reason"],
    },
    requires_confirmation=False,
    is_builtin=True,
)
