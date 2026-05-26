"""WebhookTool — universal HTTP caller for tenant API endpoints.

Each WebhookTool wraps a single HTTP endpoint on the tenant's backend.
The LLM provides arguments, and the tool makes the HTTP call.
"""
from __future__ import annotations

import ipaddress
import re
import socket
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

import asyncio as _asyncio

import httpx

from app.rag.tools.base import ToolDefinition, ToolResult

_shared_http_client: httpx.AsyncClient | None = None
_client_lock = _asyncio.Lock()


async def _get_shared_http_client() -> httpx.AsyncClient:
    global _shared_http_client
    if _shared_http_client is None:
        async with _client_lock:
            # Double-check after acquiring lock
            if _shared_http_client is None:
                _shared_http_client = httpx.AsyncClient(timeout=30.0)
    return _shared_http_client


async def close_shared_http_client() -> None:
    """Shutdown hook — call from app lifespan to close the shared client."""
    global _shared_http_client
    if _shared_http_client is not None:
        await _shared_http_client.aclose()
        _shared_http_client = None

# Private/internal IP ranges that must be blocked (SSRF prevention)
# Using ipaddress.ip_address built-in checks covers IPv4-mapped IPv6
# (e.g., ::ffff:127.0.0.1), 0.0.0.0/8, and other edge cases.
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("169.254.0.0/16"),  # AWS metadata, link-local
    ipaddress.ip_network("100.64.0.0/10"),   # Carrier-Grade NAT (cloud)
]


def _is_ip_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Check if an IP address is private, loopback, or otherwise blocked.

    Uses Python stdlib checks that handle IPv4-mapped IPv6 (::ffff:x.x.x.x),
    link-local, reserved, and unspecified addresses.
    """
    # Handle IPv4-mapped IPv6 by extracting the IPv4 part
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped

    from app.config import get_settings
    if get_settings().app_env == "development":
        return False

    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        return True
    # Additional explicit checks for cloud metadata ranges
    return any(ip in network for network in _BLOCKED_NETWORKS)


class SSRFError(Exception):
    """Raised when a URL targets a blocked internal/private address."""


async def validate_url_safety(url: str) -> str:
    """Validate a URL is not targeting internal/private resources.

    Checks:
    - Scheme is http or https (no file://, ftp://, etc.)
    - Hostname does not resolve to a private/internal IP
    - Handles IPv4-mapped IPv6 addresses

    Raises:
        SSRFError: If the URL targets a blocked resource.

    Returns:
        The validated safe IP address as a string.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SSRFError(f"Only HTTP(S) schemes allowed, got: {parsed.scheme}")

    hostname = parsed.hostname or ""
    if not hostname:
        raise SSRFError("URL has no hostname")

    # Resolve hostname to IP and check against blocked ranges without blocking the event loop
    try:
        loop = _asyncio.get_running_loop()
        addrs = await loop.getaddrinfo(
            hostname,
            None,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as e:
        # DNS resolution failed — block it as unsafe
        raise SSRFError(f"DNS resolution failed for hostname {hostname}: {e}") from e

    validated_ip: str = ""
    for _family, _type, _proto, _canonname, sockaddr in addrs:
        ip_str = sockaddr[0]
        ip = ipaddress.ip_address(ip_str)
        if _is_ip_blocked(ip):
            raise SSRFError(
                f"URL resolves to blocked private/internal address"
            )
        if not validated_ip:
            validated_ip = ip_str  # Use the first safe IP

    if not validated_ip:
        raise SSRFError("Hostname resolved to empty addresses list")
    
    return validated_ip


@dataclass(frozen=True)
class WebhookToolConfig:
    """Configuration for a single webhook tool endpoint."""

    name: str
    description: str
    http_method: str  # GET, POST, PUT, PATCH, DELETE
    endpoint_url: str  # Full or relative URL
    parameters_schema: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    timeout: float = 15.0
    response_mapping: dict[str, str] | None = None


class WebhookTool:
    """Calls an external HTTP endpoint on behalf of the tenant."""

    def __init__(
        self,
        config: WebhookToolConfig,
        *,
        auth_header: str = "",
        auth_value: str = "",
    ) -> None:
        self.config = config
        self._auth_header = auth_header or "Authorization"
        self._auth_value = auth_value  # e.g. "Bearer xxx" or API key

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.config.name,
            description=self.config.description,
            parameters=self.config.parameters_schema,
            requires_confirmation=self.config.requires_confirmation,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Call the tenant's API endpoint.

        Path parameters (e.g., ``{appointment_id}``) in the URL are
        substituted from ``arguments``.  Remaining arguments become
        query params (GET/DELETE) or JSON body (POST/PUT/PATCH).
        """
        start = time.monotonic()
        # Build URL and check for SSRF
        url = self._build_url(arguments)
        
        # SSRF protection: block requests to internal networks and prevent DNS rebinding
        try:
            safe_ip = await validate_url_safety(url)
        except SSRFError as e:
            elapsed = (time.monotonic() - start) * 1000
            return ToolResult(
                success=False, data={},
                error="URL targets a blocked address",
                execution_time_ms=elapsed,
            )

        body = self._strip_path_params(arguments)
        headers = self._build_headers()
        
        # Inject the original hostname into the Host header to prevent DNS rebinding
        parsed_url = urllib.parse.urlparse(url)
        original_hostname = parsed_url.hostname
        if original_hostname:
            headers["Host"] = original_hostname
        
        # Replace hostname in the URL with the pre-resolved safe IP
        safe_url = parsed_url._replace(
            netloc=parsed_url.netloc.replace(original_hostname, safe_ip, 1)
        ).geturl()

        method = self.config.http_method.upper()

        try:
            client = await _get_shared_http_client()
            response = await client.request(
                method=method,
                url=safe_url,
                headers=headers,
                json=body if method in ("POST", "PUT", "PATCH") else None,
                params=body if method in ("GET", "DELETE") else None,
                timeout=self.config.timeout,
            )
            elapsed = (time.monotonic() - start) * 1000

            if response.is_success:
                try:
                    data = response.json()
                except Exception:
                    data = {"raw": response.text[:2000]}
                if self.config.response_mapping:
                    data = self._apply_mapping(data)
                return ToolResult(success=True, data=data, execution_time_ms=elapsed)

            return ToolResult(
                success=False,
                data={},
                error=f"API returned HTTP {response.status_code}",
                execution_time_ms=elapsed,
            )
        except httpx.TimeoutException:
            elapsed = (time.monotonic() - start) * 1000
            return ToolResult(
                success=False,
                data={},
                error="Request timed out",
                execution_time_ms=elapsed,
            )
        except httpx.HTTPError:
            elapsed = (time.monotonic() - start) * 1000
            return ToolResult(
                success=False,
                data={},
                error="HTTP request failed",  # Generic — no internal details
                execution_time_ms=elapsed,
            )

    # ── Private helpers ──────────────────────────────────────────

    def _build_url(self, arguments: dict[str, Any]) -> str:
        """Substitute path parameters like ``{id}`` in the URL.

        Values are URL-encoded to prevent path traversal/injection.
        """
        url = self.config.endpoint_url
        path_params = re.findall(r"\{(\w+)\}", url)
        for param in path_params:
            if param in arguments:
                # URL-encode to prevent path traversal/injection
                safe_value = urllib.parse.quote(str(arguments[param]), safe="")
                url = url.replace(f"{{{param}}}", safe_value)
        return url

    def _strip_path_params(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Remove path params from arguments — they're already in the URL."""
        path_params = set(re.findall(r"\{(\w+)\}", self.config.endpoint_url))
        return {k: v for k, v in arguments.items() if k not in path_params}

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._auth_value:
            headers[self._auth_header] = self._auth_value
        return headers

    def _apply_mapping(self, data: dict) -> dict:
        """Extract specific fields from the API response."""
        if not isinstance(data, dict) or not self.config.response_mapping:
            return data
        return {
            target: data.get(source, None)
            for target, source in self.config.response_mapping.items()
        }
