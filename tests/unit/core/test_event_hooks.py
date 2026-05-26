"""Unit tests for outbound event hook dispatcher.

Covers:
    - dispatch_event: valid hook config creates task and sends HTTP POST,
      no config returns None, no event_hooks key returns None,
      no URL for event type returns None, custom headers passed through,
      EventType enum values
    - HookPayload: auto-generates timestamp, custom data dict
    - _send_hook: success, timeout (no crash), generic error (no crash)
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core.event_hooks import (
    EventType,
    HookPayload,
    _send_hook,
    dispatch_event,
)


# ── Helpers ───────────────────────────────────────────────────────


def _mock_async_client(*, post_side_effect=None, status_code=200):
    """Build a mock httpx.AsyncClient that works as an async context manager.

    Returns a tuple of (factory_callable, mock_client) so tests can inspect
    ``mock_client.post`` after awaiting the hook.
    """
    mock_client = AsyncMock()

    if post_side_effect is not None:
        mock_client.post.side_effect = post_side_effect
    else:
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_client.post.return_value = mock_response

    @asynccontextmanager
    async def _ctx(*_args, **_kwargs):
        yield mock_client

    return _ctx, mock_client


@pytest.fixture(autouse=True)
def _mock_logger():
    """Mock the structlog logger to avoid the event kwarg conflict bug.

    The source code passes ``event=payload.get("event")`` to structlog
    methods that already take ``event`` as their first positional arg.
    Mocking the logger isolates tests from this issue.
    """
    mock_log = MagicMock()
    with patch("app.core.event_hooks.logger", mock_log):
        yield mock_log


@pytest.fixture(autouse=True)
def _mock_url_safety():
    """Mock validate_url_safety to avoid real DNS resolution."""
    with patch("app.rag.tools.webhook.validate_url_safety", return_value="93.184.216.34"):
        yield


# ── HookPayload Tests ────────────────────────────────────────────


class TestHookPayload:
    """Tests for the HookPayload dataclass."""

    def test_auto_generates_timestamp(self) -> None:
        """Timestamp should be auto-populated when not provided."""
        before = datetime.now(timezone.utc).isoformat()
        payload = HookPayload(
            event="on_escalation",
            tenant_id="t1",
            conversation_id="conv-1",
        )
        after = datetime.now(timezone.utc).isoformat()
        assert payload.timestamp >= before
        assert payload.timestamp <= after

    def test_custom_timestamp_preserved(self) -> None:
        """Explicit timestamp should not be overwritten."""
        ts = "2026-01-01T00:00:00+00:00"
        payload = HookPayload(
            event="on_escalation",
            tenant_id="t1",
            conversation_id="conv-1",
            timestamp=ts,
        )
        assert payload.timestamp == ts

    def test_custom_data_dict(self) -> None:
        """Custom data dictionary should be stored correctly."""
        data = {"agent_id": "a1", "reason": "timeout"}
        payload = HookPayload(
            event="on_tool_failure",
            tenant_id="t1",
            conversation_id="conv-1",
            data=data,
        )
        assert payload.data == data
        assert payload.data["agent_id"] == "a1"

    def test_data_defaults_to_empty_dict(self) -> None:
        """Data field should default to empty dict."""
        payload = HookPayload(
            event="on_escalation",
            tenant_id="t1",
            conversation_id="conv-1",
        )
        assert payload.data == {}


# ── _send_hook Tests ─────────────────────────────────────────────


class TestSendHook:
    """Tests for the internal _send_hook HTTP POST function."""

    @pytest.mark.asyncio
    async def test_send_hook_success(self) -> None:
        """Successful POST should complete without raising."""
        ctx_factory, mock_client = _mock_async_client(status_code=200)

        with patch("app.core.event_hooks.httpx.AsyncClient", ctx_factory):
            await _send_hook(
                url="https://example.com/hook",
                payload={"event": "on_escalation", "tenant_id": "t1"},
                headers={"Content-Type": "application/json"},
            )

        mock_client.post.assert_called_once_with(
            "https://93.184.216.34/hook",
            json={"event": "on_escalation", "tenant_id": "t1"},
            headers={"Content-Type": "application/json", "Host": "example.com"},
        )

    @pytest.mark.asyncio
    async def test_send_hook_timeout_no_crash(self) -> None:
        """Timeout exception should be caught and logged, not raised."""
        ctx_factory, _ = _mock_async_client(
            post_side_effect=httpx.TimeoutException("timed out"),
        )

        with patch("app.core.event_hooks.httpx.AsyncClient", ctx_factory):
            # Should not raise
            await _send_hook(
                url="https://example.com/hook",
                payload={"event": "on_escalation"},
                headers={},
            )

    @pytest.mark.asyncio
    async def test_send_hook_generic_error_no_crash(self) -> None:
        """Generic exceptions should be caught and logged, not raised."""
        ctx_factory, _ = _mock_async_client(
            post_side_effect=ConnectionError("connection reset"),
        )

        with patch("app.core.event_hooks.httpx.AsyncClient", ctx_factory):
            # Should not raise
            await _send_hook(
                url="https://example.com/hook",
                payload={"tenant_id": "t1"},
                headers={},
            )


# ── dispatch_event Tests ─────────────────────────────────────────


class TestDispatchEvent:
    """Tests for fire-and-forget webhook dispatch."""

    def _make_payload(self) -> HookPayload:
        """Helper to create a minimal HookPayload."""
        return HookPayload(
            event="on_escalation",
            tenant_id="t1",
            conversation_id="conv-1",
        )

    @pytest.mark.asyncio
    async def test_valid_hook_config_creates_task(self) -> None:
        """Valid config with matching event URL should create a background task."""
        config = {
            "event_hooks": {
                "on_escalation": {
                    "url": "https://example.com/hooks/escalation",
                },
            },
        }

        ctx_factory, mock_client = _mock_async_client(status_code=200)

        with patch("app.core.event_hooks.httpx.AsyncClient", ctx_factory):
            task = dispatch_event(
                config, EventType.ON_ESCALATION, self._make_payload()
            )
            assert isinstance(task, asyncio.Task)
            await task  # Let it complete

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[0][0] == "https://93.184.216.34/hooks/escalation"

    @pytest.mark.asyncio
    async def test_no_config_returns_none(self) -> None:
        """None config should return None immediately."""
        result = dispatch_event(None, EventType.ON_ESCALATION, self._make_payload())
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_config_returns_none(self) -> None:
        """Empty dict config should return None."""
        result = dispatch_event({}, EventType.ON_ESCALATION, self._make_payload())
        assert result is None

    @pytest.mark.asyncio
    async def test_no_event_hooks_key_returns_none(self) -> None:
        """Config without event_hooks key should return None."""
        config = {"other_setting": "value"}
        result = dispatch_event(
            config, EventType.ON_ESCALATION, self._make_payload()
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_no_url_for_event_returns_none(self) -> None:
        """Config with event_hooks but no matching event URL should return None."""
        config = {
            "event_hooks": {
                "on_tool_failure": {
                    "url": "https://example.com/hooks/tool-failure",
                },
            },
        }
        result = dispatch_event(
            config, EventType.ON_ESCALATION, self._make_payload()
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_url_returns_none(self) -> None:
        """Hook config with empty URL string should return None."""
        config = {
            "event_hooks": {
                "on_escalation": {
                    "url": "",
                },
            },
        }
        result = dispatch_event(
            config, EventType.ON_ESCALATION, self._make_payload()
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_custom_headers_passed_through(self) -> None:
        """Custom headers from config should be merged into the POST request."""
        config = {
            "event_hooks": {
                "on_escalation": {
                    "url": "https://example.com/hooks/escalation",
                    "headers": {"X-Hook-Secret": "s3cret", "X-Custom": "val"},
                },
            },
        }

        ctx_factory, mock_client = _mock_async_client(status_code=200)

        with patch("app.core.event_hooks.httpx.AsyncClient", ctx_factory):
            task = dispatch_event(
                config, EventType.ON_ESCALATION, self._make_payload()
            )
            assert task is not None
            await task

        call_kwargs = mock_client.post.call_args
        headers_sent = call_kwargs[1]["headers"]
        assert headers_sent["X-Hook-Secret"] == "s3cret"
        assert headers_sent["X-Custom"] == "val"
        assert headers_sent["Content-Type"] == "application/json"
        assert headers_sent["User-Agent"] == "SupportForge-Hooks/1.0"
        assert headers_sent["Host"] == "example.com"

    @pytest.mark.asyncio
    async def test_event_type_enum_values(self) -> None:
        """All EventType enum values should be valid string identifiers."""
        assert EventType.ON_ESCALATION.value == "on_escalation"
        assert EventType.ON_NEW_CONVERSATION.value == "on_new_conversation"
        assert EventType.ON_TOOL_FAILURE.value == "on_tool_failure"
        assert EventType.ON_NEGATIVE_FEEDBACK.value == "on_negative_feedback"

    @pytest.mark.asyncio
    async def test_invalid_headers_type_defaults_to_empty(self) -> None:
        """Non-dict headers in config should be treated as empty dict."""
        config = {
            "event_hooks": {
                "on_escalation": {
                    "url": "https://example.com/hooks/escalation",
                    "headers": "not-a-dict",
                },
            },
        }

        ctx_factory, mock_client = _mock_async_client(status_code=200)

        with patch("app.core.event_hooks.httpx.AsyncClient", ctx_factory):
            task = dispatch_event(
                config, EventType.ON_ESCALATION, self._make_payload()
            )
            assert task is not None
            await task

        call_kwargs = mock_client.post.call_args
        headers_sent = call_kwargs[1]["headers"]
        # Only the default headers, no custom ones
        assert headers_sent == {
            "Content-Type": "application/json",
            "User-Agent": "SupportForge-Hooks/1.0",
            "Host": "example.com",
        }
