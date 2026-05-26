"""Tests for the pluggable tool system — base types, webhook, executor, resolver."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.rag.tools.base import (
    ESCALATE_TOOL_DEFINITION,
    ToolDefinition,
    ToolResult,
)
from app.rag.tools.executor import ToolExecutor


@pytest.fixture(autouse=True)
def _reset_shared_http_client():
    """Reset the shared httpx client between tests to prevent state leaks."""
    yield
    import app.rag.tools.webhook as wh
    wh._shared_http_client = None
from app.rag.tools.resolver import BuiltinEscalateTool, resolve_tenant_tools
from app.rag.tools.webhook import WebhookTool, WebhookToolConfig


# ── ToolDefinition ──────────────────────────────────────────────────


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_to_openai_format(self) -> None:
        td = ToolDefinition(
            name="check_availability",
            description="Check slot availability",
            parameters={
                "type": "object",
                "properties": {"date": {"type": "string"}},
                "required": ["date"],
            },
        )
        result = td.to_openai_format()
        assert result["type"] == "function"
        assert result["function"]["name"] == "check_availability"
        assert result["function"]["description"] == "Check slot availability"
        assert result["function"]["parameters"]["required"] == ["date"]

    def test_escalate_tool_is_builtin(self) -> None:
        assert ESCALATE_TOOL_DEFINITION.is_builtin is True
        assert ESCALATE_TOOL_DEFINITION.name == "escalate"
        fmt = ESCALATE_TOOL_DEFINITION.to_openai_format()
        assert fmt["function"]["name"] == "escalate"

    def test_default_values(self) -> None:
        td = ToolDefinition(name="test", description="desc")
        assert td.parameters == {}
        assert td.requires_confirmation is False
        assert td.is_builtin is False


# ── ToolResult ──────────────────────────────────────────────────────


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_result(self) -> None:
        r = ToolResult(success=True, data={"status": "ok"})
        assert r.success is True
        assert r.data == {"status": "ok"}
        assert r.error is None

    def test_error_result(self) -> None:
        r = ToolResult(success=False, error="Connection timeout")
        assert r.success is False
        assert r.data == {}
        assert r.error == "Connection timeout"


# ── WebhookTool ─────────────────────────────────────────────────────


class TestWebhookTool:
    """Tests for WebhookTool."""

    @pytest.fixture(autouse=True)
    def mock_url_safety(self) -> Any:
        with patch("app.rag.tools.webhook.validate_url_safety", return_value="93.184.216.34"):
            yield

    def _make_tool(self, **overrides: Any) -> WebhookTool:
        defaults = {
            "name": "get_status",
            "description": "Get order status",
            "http_method": "GET",
            "endpoint_url": "https://api.example.com/orders/{order_id}",
            "parameters_schema": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
            },
            "timeout": 10.0,
        }
        defaults.update(overrides)
        config = WebhookToolConfig(**defaults)
        return WebhookTool(config, auth_value="Bearer test-key")

    def test_definition_property(self) -> None:
        tool = self._make_tool()
        defn = tool.definition
        assert defn.name == "get_status"
        assert defn.description == "Get order status"

    @pytest.mark.asyncio
    async def test_execute_success_get(self) -> None:
        tool = self._make_tool()
        mock_response = httpx.Response(
            200,
            json={"order_id": "123", "status": "shipped"},
            request=httpx.Request("GET", "https://api.example.com/orders/123"),
        )
        with patch("httpx.AsyncClient.request", return_value=mock_response):
            result = await tool.execute({"order_id": "123"})
        assert result.success is True
        assert result.data["status"] == "shipped"
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_execute_post_with_body(self) -> None:
        tool = self._make_tool(
            http_method="POST",
            endpoint_url="https://api.example.com/bookings",
        )
        mock_response = httpx.Response(
            201,
            json={"booking_id": "b-456"},
            request=httpx.Request("POST", "https://api.example.com/bookings"),
        )
        with patch("httpx.AsyncClient.request", return_value=mock_response):
            result = await tool.execute({"date": "2026-01-15", "time": "10:00"})
        assert result.success is True
        assert result.data["booking_id"] == "b-456"

    @pytest.mark.asyncio
    async def test_execute_http_error(self) -> None:
        tool = self._make_tool()
        mock_response = httpx.Response(
            500,
            text="Internal Server Error",
            request=httpx.Request("GET", "https://api.example.com/orders/123"),
        )
        with patch("httpx.AsyncClient.request", return_value=mock_response):
            result = await tool.execute({"order_id": "123"})
        assert result.success is False
        assert "500" in result.error

    @pytest.mark.asyncio
    async def test_execute_timeout(self) -> None:
        tool = self._make_tool(timeout=0.1)
        with patch(
            "httpx.AsyncClient.request",
            side_effect=httpx.TimeoutException("Timed out"),
        ):
            result = await tool.execute({"order_id": "123"})
        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_path_param_substitution(self) -> None:
        tool = self._make_tool(
            endpoint_url="https://api.example.com/orders/{order_id}/items/{item_id}",
        )
        url = tool._build_url({"order_id": "o-123", "item_id": "i-456"})
        assert url == "https://api.example.com/orders/o-123/items/i-456"

    def test_strip_path_params(self) -> None:
        tool = self._make_tool()  # endpoint has {order_id}
        remaining = tool._strip_path_params(
            {"order_id": "123", "include_tracking": True}
        )
        assert "order_id" not in remaining
        assert remaining["include_tracking"] is True

    def test_response_mapping(self) -> None:
        tool = self._make_tool(
            response_mapping={"available": "is_available", "time": "next_slot"},
        )
        config = tool.config
        mapped = tool._apply_mapping({"is_available": True, "next_slot": "10:00"})
        assert mapped == {"available": True, "time": "10:00"}

    def test_auth_header(self) -> None:
        tool = self._make_tool()
        headers = tool._build_headers()
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_execute_non_json_response(self) -> None:
        """API returns HTML/plain text — verify {"raw": text[:2000]} fallback."""
        tool = self._make_tool()
        html_body = "<html><body>Not JSON</body></html>"
        mock_response = httpx.Response(
            200,
            text=html_body,
            headers={"Content-Type": "text/html"},
            request=httpx.Request("GET", "https://api.example.com/orders/123"),
        )
        with patch("httpx.AsyncClient.request", return_value=mock_response):
            result = await tool.execute({"order_id": "123"})
        assert result.success is True
        assert result.data == {"raw": html_body}

    def test_no_auth_header_when_auth_empty(self) -> None:
        """When auth_value is empty, Authorization header should NOT be in headers."""
        config = WebhookToolConfig(
            name="no_auth",
            description="Tool without auth",
            http_method="GET",
            endpoint_url="https://api.example.com/public",
        )
        tool = WebhookTool(config, auth_value="")
        headers = tool._build_headers()
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_build_url_missing_path_param(self) -> None:
        """Path param placeholder left unsubstituted when arg missing."""
        tool = self._make_tool(
            endpoint_url="https://api.example.com/orders/{order_id}/items/{item_id}",
        )
        # Only provide order_id, not item_id
        url = tool._build_url({"order_id": "o-123"})
        assert "o-123" in url
        # item_id placeholder should remain since the argument was missing
        assert "{item_id}" in url


# ── ToolExecutor ────────────────────────────────────────────────────


class TestToolExecutor:
    """Tests for ToolExecutor safety guardrails."""

    @pytest.mark.asyncio
    async def test_execute_success(self) -> None:
        executor = ToolExecutor(max_rounds=3)
        mock_tool = MagicMock()
        mock_tool.config.name = "test_tool"
        mock_tool.config.timeout = 10.0
        mock_tool.definition.is_builtin = False
        mock_tool.execute = AsyncMock(
            return_value=ToolResult(success=True, data={"ok": True})
        )
        result = await executor.execute(mock_tool, {"key": "value"})
        assert result.success is True
        assert result.data == {"ok": True}

    @pytest.mark.asyncio
    async def test_execute_error_isolation(self) -> None:
        executor = ToolExecutor()
        mock_tool = MagicMock()
        mock_tool.config.name = "failing_tool"
        mock_tool.config.timeout = 10.0
        mock_tool.definition.is_builtin = False
        mock_tool.execute = AsyncMock(side_effect=RuntimeError("Connection failed"))
        result = await executor.execute(mock_tool, {})
        assert result.success is False
        assert "Tool execution failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_builtin_escalate(self) -> None:
        executor = ToolExecutor()
        escalate = BuiltinEscalateTool()
        result = await executor.execute(escalate, {"reason": "Customer angry"})
        assert result.success is True
        assert result.data["escalated"] is True
        assert result.data["reason"] == "Customer angry"

    @pytest.mark.asyncio
    async def test_response_truncation(self) -> None:
        executor = ToolExecutor()
        # Make a response larger than MAX_RESPONSE_BYTES
        big_data = {"content": "x" * 60_000}
        mock_tool = MagicMock()
        mock_tool.config.name = "big_tool"
        mock_tool.config.timeout = 10.0
        mock_tool.definition.is_builtin = False
        mock_tool.execute = AsyncMock(
            return_value=ToolResult(success=True, data=big_data)
        )
        result = await executor.execute(mock_tool, {})
        assert result.success is True
        assert result.data.get("_truncated") is True

    def test_default_max_rounds(self) -> None:
        executor = ToolExecutor()
        assert executor.max_rounds == 3

    @pytest.mark.asyncio
    async def test_executor_asyncio_timeout(self) -> None:
        """asyncio.wait_for timeout (separate from httpx timeout)."""
        executor = ToolExecutor()
        mock_tool = MagicMock()
        mock_tool.config.name = "slow_tool"
        mock_tool.config.timeout = 0.01  # Very short timeout
        mock_tool.definition.is_builtin = False

        async def slow_execute(args: dict) -> ToolResult:
            await asyncio.sleep(10)  # Much longer than timeout
            return ToolResult(success=True, data={})  # pragma: no cover

        mock_tool.execute = slow_execute
        result = await executor.execute(mock_tool, {})
        assert result.success is False
        assert "timed out" in result.error.lower()


# ── resolve_tenant_tools ────────────────────────────────────────────


class TestResolveTenantTools:
    """Tests for resolve_tenant_tools."""

    def test_none_config_returns_only_escalate(self) -> None:
        tools = resolve_tenant_tools(None)
        assert len(tools) == 1
        assert tools[0].definition.name == "escalate"

    def test_empty_config_returns_only_escalate(self) -> None:
        tools = resolve_tenant_tools({})
        assert len(tools) == 1
        assert tools[0].definition.name == "escalate"

    def test_tools_disabled_returns_only_escalate(self) -> None:
        config = {"tools_enabled": False, "tools": [{"name": "x", "description": "y"}]}
        tools = resolve_tenant_tools(config)
        assert len(tools) == 1

    def test_medforge_config(self) -> None:
        config = {
            "tools_enabled": True,
            "tools_base_url": "https://medforge-api.example.com",
            "tools": [
                {
                    "name": "check_availability",
                    "description": "Check doctor availability",
                    "http_method": "GET",
                    "endpoint": "/api/slots",
                    "parameters": {
                        "type": "object",
                        "properties": {"date": {"type": "string"}},
                    },
                },
                {
                    "name": "book_appointment",
                    "description": "Book an appointment",
                    "http_method": "POST",
                    "endpoint": "/api/bookings",
                    "timeout": 20.0,
                },
            ],
        }
        secrets = {"tools_auth.default": "Bearer api-key-123"}
        tools = resolve_tenant_tools(config, secrets=secrets)

        # 2 tenant tools + 1 escalate
        assert len(tools) == 3
        names = [t.definition.name for t in tools]
        assert "check_availability" in names
        assert "book_appointment" in names
        assert "escalate" in names
        # Escalate should be last
        assert tools[-1].definition.name == "escalate"

    def test_per_tool_auth(self) -> None:
        config = {
            "tools_enabled": True,
            "tools_base_url": "https://api.example.com",
            "tools": [{"name": "special", "description": "desc", "endpoint": "/special"}],
        }
        secrets = {
            "tools_auth.default": "Bearer default-key",
            "tools_auth.special": "Bearer special-key",
        }
        tools = resolve_tenant_tools(config, secrets=secrets)
        # First tool is 'special', should have per-tool auth
        webhook_tool = tools[0]
        assert webhook_tool._auth_value == "Bearer special-key"

    def test_invalid_tool_entry_skipped(self) -> None:
        config = {
            "tools_enabled": True,
            "tools": [
                {"name": "good", "description": "desc", "endpoint": "/good"},
                {"bad": "no name or description"},
                42,
            ],
        }
        tools = resolve_tenant_tools(config)
        # Only 'good' + escalate
        assert len(tools) == 2
        assert tools[0].definition.name == "good"

    def test_base_url_prepended(self) -> None:
        config = {
            "tools_enabled": True,
            "tools_base_url": "https://api.example.com",
            "tools": [{"name": "t", "description": "d", "endpoint": "/v1/stuff"}],
        }
        tools = resolve_tenant_tools(config)
        assert tools[0].config.endpoint_url == "https://api.example.com/v1/stuff"

    def test_absolute_url_not_prepended(self) -> None:
        config = {
            "tools_enabled": True,
            "tools_base_url": "https://api.example.com",
            "tools": [
                {
                    "name": "t",
                    "description": "d",
                    "endpoint": "https://other-api.example.com/v1/stuff",
                }
            ],
        }
        tools = resolve_tenant_tools(config)
        assert tools[0].config.endpoint_url == "https://other-api.example.com/v1/stuff"
