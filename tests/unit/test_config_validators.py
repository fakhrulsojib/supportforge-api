"""Tests for tenant config_json validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config_validators import (
    TenantConfigValidator,
    ToolConfigItem,
    validate_config_json,
)
from app.rag.tools.webhook import WebhookTool, WebhookToolConfig


class TestToolConfigItem:
    """Tests for ToolConfigItem validation."""

    def test_valid_tool(self) -> None:
        item = ToolConfigItem(
            name="check_status",
            description="Check order status",
            http_method="GET",
            endpoint="/api/orders/{id}",
        )
        assert item.name == "check_status"
        assert item.http_method == "GET"

    def test_http_method_normalized_uppercase(self) -> None:
        item = ToolConfigItem(name="t", description="d", endpoint="/x", http_method="post")
        assert item.http_method == "POST"

    def test_invalid_http_method(self) -> None:
        with pytest.raises(ValidationError, match="http_method"):
            ToolConfigItem(name="t", description="d", endpoint="/x", http_method="INVALID")

    def test_timeout_too_large(self) -> None:
        with pytest.raises(ValidationError, match="timeout"):
            ToolConfigItem(name="t", description="d", endpoint="/x", timeout=200)

    def test_timeout_zero(self) -> None:
        with pytest.raises(ValidationError, match="timeout"):
            ToolConfigItem(name="t", description="d", endpoint="/x", timeout=0)

    def test_name_too_long(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            ToolConfigItem(name="a" * 101, description="d", endpoint="/x")

    def test_name_empty(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            ToolConfigItem(name="", description="d", endpoint="/x")


class TestTenantConfigValidator:
    """Tests for TenantConfigValidator."""

    def test_valid_full_config(self) -> None:
        config = TenantConfigValidator(
            tools_enabled=True,
            tools_base_url="https://api.example.com",
            max_tool_rounds=5,
            tools=[
                ToolConfigItem(name="t1", description="d1", endpoint="/x"),
                ToolConfigItem(name="t2", description="d2", endpoint="/y"),
            ],
        )
        assert config.tools_enabled is True
        assert config.max_tool_rounds == 5
        assert len(config.tools) == 2

    def test_extra_keys_allowed(self) -> None:
        """Backward compat: unknown keys should pass through."""
        config = TenantConfigValidator(
            chat_model="gpt-4",  # type: ignore[call-arg]
            temperature=0.5,  # type: ignore[call-arg]
        )
        assert config.model_extra.get("chat_model") == "gpt-4"

    def test_max_tool_rounds_too_high(self) -> None:
        with pytest.raises(ValidationError, match="max_tool_rounds"):
            TenantConfigValidator(max_tool_rounds=11)

    def test_max_tool_rounds_too_low(self) -> None:
        with pytest.raises(ValidationError, match="max_tool_rounds"):
            TenantConfigValidator(max_tool_rounds=0)

    def test_duplicate_tool_names(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            TenantConfigValidator(
                tools=[
                    ToolConfigItem(name="same", description="d1", endpoint="/x"),
                    ToolConfigItem(name="same", description="d2", endpoint="/y"),
                ],
            )


class TestValidateConfigJson:
    """Tests for the validate_config_json function."""

    def test_valid_config_roundtrips(self) -> None:
        config = {
            "tools_enabled": True,
            "tools": [{"name": "t", "description": "d", "endpoint": "/x"}],
            "chat_model": "gemini-2.5-flash",
        }
        result = validate_config_json(config)
        assert result["tools_enabled"] is True
        assert result["chat_model"] == "gemini-2.5-flash"
        assert len(result["tools"]) == 1

    def test_invalid_config_raises(self) -> None:
        config = {
            "tools": [{"name": "", "description": "d", "endpoint": "/x"}],
        }
        with pytest.raises(ValidationError):
            validate_config_json(config)

    def test_empty_config_passes(self) -> None:
        result = validate_config_json({})
        assert isinstance(result, dict)

    def test_none_tools_passes(self) -> None:
        result = validate_config_json({"tools": None})
        assert result.get("tools") is None

    def test_max_tool_count_exceeded(self) -> None:
        """Maximum 20 tools per tenant."""
        tools = [
            {"name": f"tool_{i}", "description": "d", "endpoint": "/x"}
            for i in range(21)
        ]
        with pytest.raises(ValidationError, match="Maximum 20"):
            validate_config_json({"tools": tools})

    def test_endpoint_file_scheme_rejected(self) -> None:
        """file:// scheme must be blocked (SSRF prevention)."""
        with pytest.raises(ValidationError, match="Only HTTP"):
            ToolConfigItem(
                name="bad", description="d",
                endpoint="file:///etc/passwd",
            )

    def test_endpoint_ftp_scheme_rejected(self) -> None:
        """ftp:// scheme must be blocked."""
        with pytest.raises(ValidationError, match="Only HTTP"):
            ToolConfigItem(
                name="bad", description="d",
                endpoint="ftp://internal/data",
            )

    def test_endpoint_https_allowed(self) -> None:
        """HTTPS endpoints pass validation."""
        item = ToolConfigItem(
            name="ok", description="d",
            endpoint="https://api.example.com/v1",
        )
        assert item.endpoint.startswith("https://")

    def test_endpoint_relative_path_allowed(self) -> None:
        """Relative paths (no scheme) are valid — base_url will be prepended."""
        item = ToolConfigItem(
            name="ok", description="d",
            endpoint="/api/v1/check",
        )
        assert item.endpoint == "/api/v1/check"

    def test_tools_base_url_non_http_rejected(self) -> None:
        """Non-HTTP(S) base URL scheme must be blocked."""
        with pytest.raises(ValidationError, match="Only HTTP"):
            TenantConfigValidator(
                tools_base_url="file:///tmp",
            )


class TestSSRFProtection:
    """Tests for SSRF protection in webhook tool."""

    @pytest.mark.asyncio
    async def test_validate_url_blocks_file_scheme(self) -> None:
        from app.rag.tools.webhook import SSRFError, validate_url_safety

        with pytest.raises(SSRFError, match="Only HTTP"):
            await validate_url_safety("file:///etc/passwd")

    @pytest.mark.asyncio
    async def test_validate_url_blocks_ftp_scheme(self) -> None:
        from app.rag.tools.webhook import SSRFError, validate_url_safety

        with pytest.raises(SSRFError, match="Only HTTP"):
            await validate_url_safety("ftp://internal/data")

    @pytest.mark.asyncio
    async def test_validate_url_blocks_no_hostname(self) -> None:
        from app.rag.tools.webhook import SSRFError, validate_url_safety

        with pytest.raises(SSRFError, match="no hostname"):
            await validate_url_safety("http://")

    @pytest.mark.asyncio
    async def test_validate_url_blocks_localhost(self) -> None:
        from app.rag.tools.webhook import SSRFError, validate_url_safety

        with pytest.raises(SSRFError, match="blocked"):
            await validate_url_safety("http://127.0.0.1/admin")

    @pytest.mark.asyncio
    async def test_validate_url_allows_public_https(self) -> None:
        from unittest.mock import AsyncMock, patch
        from app.rag.tools.webhook import validate_url_safety

        # Mock DNS resolution to return a safe public IP
        mock_addrs = [
            (2, 1, 6, "", ("93.184.216.34", 0)),  # AF_INET, SOCK_STREAM
        ]
        with patch("asyncio.get_running_loop") as mock_loop_fn:
            mock_loop = mock_loop_fn.return_value
            mock_loop.getaddrinfo = AsyncMock(return_value=mock_addrs)
            result = await validate_url_safety("https://api.example.com/v1/check")
        assert result == "93.184.216.34"

    @pytest.mark.asyncio
    async def test_webhook_blocks_internal_url(self) -> None:
        """WebhookTool should return error for internal URLs."""
        tool = WebhookTool(
            WebhookToolConfig(
                name="test",
                description="test",
                http_method="GET",
                endpoint_url="http://169.254.169.254/latest/meta-data/",
            ),
        )
        result = await tool.execute({})
        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_validate_url_blocks_ipv4_mapped_ipv6(self) -> None:
        """IPv4-mapped IPv6 loopback must be blocked (::ffff:127.0.0.1)."""
        from app.rag.tools.webhook import _is_ip_blocked

        import ipaddress

        # IPv4-mapped IPv6 loopback
        ip = ipaddress.ip_address("::ffff:127.0.0.1")
        assert _is_ip_blocked(ip) is True

        # IPv4-mapped IPv6 private
        ip = ipaddress.ip_address("::ffff:10.0.0.1")
        assert _is_ip_blocked(ip) is True

        # Regular public IPv4 should be allowed
        ip = ipaddress.ip_address("8.8.8.8")
        assert _is_ip_blocked(ip) is False


class TestValidateConfigJsonEdgeCases:
    """Edge case tests for validate_config_json."""

    def test_non_dict_raises_type_error(self) -> None:
        """Passing None/list/string should raise TypeError, not crash."""
        with pytest.raises(TypeError, match="must be a dict"):
            validate_config_json(None)  # type: ignore[arg-type]

    def test_list_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="must be a dict"):
            validate_config_json([{"tools": []}])  # type: ignore[arg-type]

    def test_string_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="must be a dict"):
            validate_config_json("invalid")  # type: ignore[arg-type]


class TestSafeJsonDumps:
    """Tests for _safe_json_dumps in tool_loop."""

    def test_normal_dict(self) -> None:
        from app.rag.tools.tool_loop import _safe_json_dumps

        result = _safe_json_dumps({"key": "value", "num": 42})
        assert result == '{"key": "value", "num": 42}'

    def test_non_serializable_falls_back(self) -> None:
        """Non-serializable objects should be converted to str, not crash."""
        from datetime import datetime

        from app.rag.tools.tool_loop import _safe_json_dumps

        result = _safe_json_dumps({"time": datetime(2026, 1, 1)})
        # Should not crash — falls back to str()
        assert isinstance(result, str)

    def test_truncation_large_payload(self) -> None:
        """Payloads >100KB should be truncated to prevent OOM."""
        from app.rag.tools.tool_loop import _safe_json_dumps

        # Create a dict whose JSON exceeds 100KB
        large_data = {"data": "x" * 120_000}
        result = _safe_json_dumps(large_data)
        assert len(result) < 120_000
        assert result.endswith("... [TRUNCATED]")

    def test_no_truncation_at_boundary(self) -> None:
        """Payloads at exactly 100KB should NOT be truncated."""
        from app.rag.tools.tool_loop import _safe_json_dumps

        # Create a small dict well under the limit
        small_data = {"key": "value"}
        result = _safe_json_dumps(small_data)
        assert "TRUNCATED" not in result

