"""Outbound event hook dispatcher.

Sends fire-and-forget HTTP POST notifications to tenant-configured
webhook URLs when platform events occur (escalation, tool failure,
new conversation, negative feedback).

Hooks are configured in ``config_json.event_hooks``::

    {
        "event_hooks": {
            "on_escalation": {
                "url": "https://api.medforge.com/hooks/escalation",
                "headers": {"X-Hook-Secret": "..."}
            },
            "on_tool_failure": {
                "url": "https://api.medforge.com/hooks/tool-failure"
            }
        }
    }

Design principles:
    - **Non-blocking**: Dispatched via ``asyncio.create_task`` — never
      delays the response to the user.
    - **Fail-safe**: Webhook failures are logged but never raised.
    - **No PII**: Payloads contain IDs and metadata, never message content.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import httpx
import structlog

# Keep strong references to background tasks to prevent garbage collection
_pending_tasks: set[asyncio.Task[Any]] = set()

logger = structlog.get_logger(__name__)

# Timeout for outbound webhook calls
_HOOK_TIMEOUT = 10.0


class EventType(str, Enum):
    """Platform events that can trigger outbound webhooks."""

    ON_ESCALATION = "on_escalation"
    ON_NEW_CONVERSATION = "on_new_conversation"
    ON_TOOL_FAILURE = "on_tool_failure"
    ON_NEGATIVE_FEEDBACK = "on_negative_feedback"


@dataclass
class HookPayload:
    """Payload sent to tenant webhook endpoints.

    Contains only IDs and metadata — never raw message content.
    """

    event: str
    tenant_id: str
    conversation_id: str
    timestamp: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


async def _send_hook(url: str, payload: dict[str, Any], headers: dict[str, str]) -> None:
    """Send a single webhook POST request.

    This is the actual HTTP call — runs inside a background task.
    All exceptions are caught and logged, never propagated.
    """
    try:
        from urllib.parse import urlparse
        from app.rag.tools.webhook import SSRFError, validate_url_safety
        
        # SSRF Protection: Resolve safe IP and prevent DNS rebinding
        safe_ip = await validate_url_safety(url)
        parsed = urlparse(url)
        original_hostname = parsed.hostname
        if original_hostname:
            headers["Host"] = original_hostname
            port_part = f":{parsed.port}" if parsed.port else ""
            # Strip auth credentials and rebuild safely
            safe_url = f"{parsed.scheme}://{safe_ip}{port_part}{parsed.path}"
            if parsed.query:
                safe_url += f"?{parsed.query}"
        else:
            safe_url = url  # validate_url_safety would have raised, but be safe

        async with httpx.AsyncClient(timeout=_HOOK_TIMEOUT, verify=False) as client:
            response = await client.post(safe_url, json=payload, headers=headers)
        logger.info(
            "event_hook_dispatched",
            event_type=payload.get("event"),
            url=url,
            status=response.status_code,
        )
    except httpx.TimeoutException:
        logger.warning(
            "event_hook_timeout",
            event_type=payload.get("event"),
            url=url,
        )
    except Exception:
        # Also catches SSRFError
        logger.warning(
            "event_hook_failed",
            event_type=payload.get("event"),
            url=url,
            exc_info=False,
        )


def dispatch_event(
    tenant_config_json: dict[str, Any] | None,
    event_type: EventType,
    payload: HookPayload,
) -> asyncio.Task[None] | None:
    """Fire-and-forget webhook dispatch for a platform event.

    Looks up the tenant's ``event_hooks`` config for the given event
    type. If a URL is registered, schedules a background HTTP POST.

    Args:
        tenant_config_json: The tenant's full ``config_json``.
        event_type: The event that occurred.
        payload: Event data to send.

    Returns:
        The background task if a hook was dispatched, None otherwise.
    """
    if not tenant_config_json:
        return None

    hooks = tenant_config_json.get("event_hooks")
    if not hooks or not isinstance(hooks, dict):
        return None

    hook_config = hooks.get(event_type.value)
    if not hook_config or not isinstance(hook_config, dict):
        return None

    url = hook_config.get("url", "")
    if not url:
        return None

    custom_headers = hook_config.get("headers", {})
    if not isinstance(custom_headers, dict):
        custom_headers = {}

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "SupportForge-Hooks/1.0",
        **custom_headers,
    }

    # Auto-set the event field to ensure caller can't diverge from event_type
    payload.event = event_type.value

    payload_dict = asdict(payload)

    try:
        task = asyncio.create_task(_send_hook(url, payload_dict, headers))
        _pending_tasks.add(task)
        task.add_done_callback(_pending_tasks.discard)
        return task
    except RuntimeError:
        # No running event loop (e.g., during shutdown or sync context)
        logger.warning(
            "event_hook_no_event_loop",
            event_type=event_type.value,
            url=url,
        )
        return None
