"""Widget API endpoints for the embeddable SDK.

Public-facing endpoints that allow anonymous visitors on tenant
websites to create chat sessions and retrieve UI configuration.

Endpoints:
    POST /api/v1/widget/session     — Create anonymous session token
    GET  /api/v1/widget/ui-config/{slug} — Get tenant UI config (public)
"""

from __future__ import annotations

import hmac
import time
from collections import defaultdict
from datetime import timedelta
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Path, Request

from app.api.schemas.widget import (
    WidgetSessionRequest,
    WidgetSessionResponse,
    WidgetUIConfigResponse,
)
from app.config import get_settings
from app.core.widget_token import create_widget_token
from app.domain.models.enums import TenantStatus
from app.infrastructure.database.connection import AsyncSessionLocal
from app.infrastructure.database.repositories.tenant_repo import SQLTenantRepository

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/widget", tags=["widget"])


# ── In-memory rate limiter (per-IP, for widget session creation) ──
# Keys: IP address → list of request timestamps
# Cleaned up lazily on each request.
# NOTE: This is per-process — with multiple workers (gunicorn --workers N),
# the effective limit is N × max_requests. For distributed deployments,
# replace with Redis-based rate limiting.
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(client_ip: str, max_requests: int, window_seconds: int = 60) -> bool:
    """Check if the client IP has exceeded the rate limit.

    Returns True if the request is allowed, False if rate-limited.
    Uses a sliding window counter with lazy cleanup.
    """
    now = time.monotonic()
    cutoff = now - window_seconds

    # Remove expired entries
    timestamps = _rate_limit_store[client_ip]
    _rate_limit_store[client_ip] = [t for t in timestamps if t > cutoff]

    # Prune empty entries to prevent unbounded memory growth
    if not _rate_limit_store[client_ip]:
        del _rate_limit_store[client_ip]
        # Re-check: always allow first request after cleanup
        _rate_limit_store[client_ip].append(now)
        return True

    if len(_rate_limit_store[client_ip]) >= max_requests:
        return False

    _rate_limit_store[client_ip].append(now)
    return True


def _get_client_ip(request: Request) -> str:
    """Extract client IP from the request.

    Checks X-Forwarded-For first (for reverse proxies), then falls
    back to the direct client IP.
    """
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        # Take the first IP in the chain (original client)
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _matches_domain(origin: str, allowed_domains: list[str]) -> bool:
    """Check if an Origin header matches the allowed domains list.

    Supports exact match and wildcard subdomains (e.g., ``*.medforge.com``).
    Origin format: ``https://www.medforge.com`` → extracted domain: ``www.medforge.com``

    Args:
        origin: The Origin header value (e.g., ``https://www.medforge.com``).
        allowed_domains: List of allowed domain patterns.

    Returns:
        True if the origin matches any allowed domain.
    """
    if not allowed_domains:
        return True  # No restriction if no domains configured
    if not origin:
        return False  # Require Origin header when domain restriction is active

    # Extract hostname from origin URL
    # "https://www.medforge.com" → "www.medforge.com"
    hostname = origin.lower()
    for prefix in ("https://", "http://"):
        if hostname.startswith(prefix):
            hostname = hostname[len(prefix):]
            break
    # Remove port if present
    hostname = hostname.split(":")[0].rstrip("/")

    for pattern in allowed_domains:
        pattern = pattern.lower().strip()
        if pattern.startswith("*."):
            # Wildcard: *.medforge.com matches sub.medforge.com and medforge.com
            base = pattern[2:]
            if hostname == base or hostname.endswith(f".{base}"):
                return True
        elif hostname == pattern:
            return True

    return False


@router.post(
    "/session",
    response_model=WidgetSessionResponse,
    status_code=200,
    summary="Create widget session",
    description="Create an anonymous session for the embeddable chat widget. "
    "Validates the embed key against the tenant's configuration.",
)
async def create_widget_session(
    request: Request,
    body: WidgetSessionRequest,
) -> WidgetSessionResponse:
    """Create an anonymous widget session token.

    Validates:
        1. Tenant exists and is active (by slug lookup)
        2. Embed key matches ``config_json.embed_key``
        3. Origin matches ``config_json.embed_domains`` (if configured)
        4. IP rate limit not exceeded
    """
    settings = get_settings()

    # Rate limit check
    client_ip = _get_client_ip(request)
    if not _check_rate_limit(client_ip, max_requests=settings.widget_rate_limit_per_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many session requests. Please try again later.",
        )

    # Look up tenant by slug
    async with AsyncSessionLocal() as session:
        tenant_repo = SQLTenantRepository(session)
        tenant = await tenant_repo.get_by_slug(body.tenant_slug)

    if not tenant:
        raise HTTPException(status_code=403, detail="Invalid tenant or embed key")

    if tenant.status != TenantStatus.ACTIVE:
        raise HTTPException(status_code=403, detail="Tenant not active")

    # Validate embed key
    config = tenant.config_json or {}
    stored_embed_key = config.get("embed_key", "")
    if not stored_embed_key or not hmac.compare_digest(stored_embed_key, body.embed_key):
        # Use same error message as "tenant not found" to avoid leaking info
        raise HTTPException(status_code=403, detail="Invalid tenant or embed key")

    # Validate Origin against embed_domains (if configured)
    origin = request.headers.get("Origin", "")
    embed_domains: list[str] = config.get("embed_domains", [])
    if embed_domains and not _matches_domain(origin, embed_domains):
        logger.warning(
            "widget_origin_rejected",
            origin=origin,
            tenant_slug=body.tenant_slug,
            allowed_domains=embed_domains,
        )
        raise HTTPException(status_code=403, detail="Origin not allowed for this embed key")

    # Create widget session token
    expires_minutes = settings.widget_session_expire_minutes
    token = create_widget_token(
        tenant_id=tenant.id,
        tenant_slug=tenant.slug,
        secret_key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
        visitor_id=body.visitor_id,
        expires_delta=timedelta(minutes=expires_minutes),
    )

    logger.info(
        "widget_session_created",
        tenant_slug=body.tenant_slug,
        visitor_id=body.visitor_id or "(anonymous)",
    )

    return WidgetSessionResponse(
        session_token=token,
        tenant_id=tenant.id,
        tenant_slug=tenant.slug,
        expires_in=expires_minutes * 60,
    )


@router.get(
    "/ui-config/{tenant_slug}",
    response_model=WidgetUIConfigResponse,
    summary="Get widget UI config",
    description="Returns public-facing UI configuration for the tenant's "
    "embeddable widget. No authentication required.",
)
async def get_widget_ui_config(
    tenant_slug: str = Path(
        ..., min_length=2, max_length=63, description="Tenant slug"
    ),
) -> WidgetUIConfigResponse:
    """Return the tenant's public UI config for the widget.

    Only exposes ``ui_config`` from ``config_json`` — never returns
    tools, secrets, agent_prompt, or other internal configuration.
    """
    async with AsyncSessionLocal() as session:
        tenant_repo = SQLTenantRepository(session)
        tenant = await tenant_repo.get_by_slug(tenant_slug)

    if not tenant or tenant.status != TenantStatus.ACTIVE:
        raise HTTPException(status_code=404, detail="Tenant not found")

    config = tenant.config_json or {}
    ui_config: dict[str, Any] = config.get("ui_config", {})

    return WidgetUIConfigResponse(
        brand_name=ui_config.get("brand_name", "Support"),
        logo_url=ui_config.get("logo_url", ""),
        welcome_message=ui_config.get("welcome_message", "Hi! How can I help you today?"),
        placeholder_text=ui_config.get("placeholder_text", "Type your message..."),
        theme=ui_config.get("theme", {}),
        widget=ui_config.get("widget", {}),
    )
