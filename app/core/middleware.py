"""Middleware for CORS, request-ID injection, and tenant context extraction."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

if TYPE_CHECKING:
    from fastapi import FastAPI
    from starlette.requests import Request
    from starlette.responses import Response

logger = structlog.get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Injects a unique X-Request-ID header into every request/response.

    If the client sends an X-Request-ID, it is preserved.
    Otherwise a new UUID is generated.
    The request ID is bound to the structlog context for correlation.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        # Store in request state for downstream access
        request.state.request_id = request_id

        # Bind to structlog context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TenantContextMiddleware(BaseHTTPMiddleware):
    """Extracts X-Tenant-ID header and injects into request state.

    Routes that require tenant context can access it via
    ``request.state.tenant_id``. If the header is missing,
    tenant_id is set to None — individual routes decide
    whether to reject the request.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        tenant_id = request.headers.get("X-Tenant-ID")
        request.state.tenant_id = tenant_id

        if tenant_id:
            structlog.contextvars.bind_contextvars(tenant_id=tenant_id)

        return await call_next(request)


# ── Dynamic CORS for widget routes ───────────────────────────────

# Paths that should use dynamic tenant-based CORS
_WIDGET_PATH_PREFIXES = ("/api/v1/widget/", "/api/v1/ws/")

# In-memory cache: { frozenset of allowed origins } with timestamp
_tenant_origins_cache: dict[str, object] = {
    "origins": set(),
    "updated_at": 0.0,
}
_CACHE_TTL_SECONDS = 60


async def _load_tenant_origins() -> set[str]:
    """Load all active tenants' embed_domains from the DB.

    Returns a set of fully-qualified origins (e.g. ``http://example.com``).
    Domains without a scheme get both ``http://`` and ``https://`` prefixed.
    """
    from app.infrastructure.database.connection import AsyncSessionLocal

    origins: set[str] = set()
    try:
        from sqlalchemy import text

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text(
                    "SELECT config_json->'embed_domains' FROM tenants "
                    "WHERE status = 'active' "
                    "AND config_json->'embed_domains' IS NOT NULL"
                )
            )
            for (domains_json,) in result:
                if not domains_json:
                    continue
                # domains_json is a list of strings from JSONB
                import json
                if isinstance(domains_json, str):
                    domains_json = json.loads(domains_json)
                for domain in domains_json:
                    domain = domain.strip().lower()
                    if not domain:
                        continue
                    if domain.startswith("http://") or domain.startswith("https://"):
                        origins.add(domain.rstrip("/"))
                    else:
                        # Bare domain → allow both schemes and common ports
                        origins.add(f"http://{domain}")
                        origins.add(f"https://{domain}")
                        # Also add with common dev ports
                        if ":" not in domain:
                            for port in (3000, 4000, 5000, 8080):
                                origins.add(f"http://{domain}:{port}")
    except Exception:
        logger.warning("tenant_origins_load_failed", exc_info=True)

    return origins


async def _get_allowed_widget_origins() -> set[str]:
    """Return cached set of allowed widget origins, refreshing if stale."""
    now = time.monotonic()
    if now - _tenant_origins_cache["updated_at"] > _CACHE_TTL_SECONDS:
        origins = await _load_tenant_origins()
        _tenant_origins_cache["origins"] = origins
        _tenant_origins_cache["updated_at"] = now
        logger.debug("tenant_origins_cache_refreshed", count=len(origins))
    return _tenant_origins_cache["origins"]


def _is_widget_path(path: str) -> bool:
    """Check if the request path is a widget/WebSocket route."""
    return any(path.startswith(prefix) for prefix in _WIDGET_PATH_PREFIXES)


class DynamicCORSMiddleware(BaseHTTPMiddleware):
    """CORS middleware with dynamic origin resolution for widget routes.

    - **Admin UI routes**: Uses a static origins list from settings.
    - **Widget routes** (``/api/v1/widget/``, ``/api/v1/ws/``): Checks
      the ``Origin`` header against all active tenants' ``embed_domains``
      (loaded from the DB and cached with a 60-second TTL).

    This allows tenants to self-serve domain allowlisting via the
    Settings UI without requiring server restarts.
    """

    def __init__(self, app, static_origins: list[str], expose_headers: list[str] | None = None):
        super().__init__(app)
        self.static_origins = set(o.rstrip("/") for o in static_origins)
        self.expose_headers = expose_headers or []

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        origin = request.headers.get("Origin", "")
        path = request.url.path

        # Determine if origin is allowed
        is_allowed = False
        if origin:
            origin_clean = origin.rstrip("/")
            # Static origins (admin UI) — always checked
            if origin_clean in self.static_origins:
                is_allowed = True
            # Widget routes — also check dynamic tenant origins
            elif _is_widget_path(path):
                widget_origins = await _get_allowed_widget_origins()
                if origin_clean in widget_origins:
                    is_allowed = True

        # Handle preflight OPTIONS
        if request.method == "OPTIONS":
            if is_allowed:
                return self._preflight_response(origin)
            else:
                # Still respond to OPTIONS but without CORS headers
                from starlette.responses import Response as StarletteResponse
                return StarletteResponse(status_code=204)

        # Handle actual request
        response = await call_next(request)
        if is_allowed and origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            if self.expose_headers:
                response.headers["Access-Control-Expose-Headers"] = ", ".join(self.expose_headers)
        return response

    def _preflight_response(self, origin: str):
        """Build a 204 preflight response with full CORS headers."""
        from starlette.responses import Response as StarletteResponse

        headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type, X-Request-ID, X-Tenant-ID",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": "600",
        }
        if self.expose_headers:
            headers["Access-Control-Expose-Headers"] = ", ".join(self.expose_headers)
        return StarletteResponse(status_code=204, headers=headers)


def setup_middleware(app: FastAPI) -> None:
    """Register all middleware on the FastAPI application.

    Order matters — middleware is executed in reverse registration order.
    We register TenantContext first (executes last), then RequestID
    (executes first), so request-ID is available for tenant logging.
    """
    from app.config import get_settings

    settings = get_settings()

    # Dynamic CORS — handles both static admin UI origins and
    # dynamic tenant widget origins from the database.
    static_origins = settings.cors_origin_list + settings.widget_cors_origin_list
    app.add_middleware(
        DynamicCORSMiddleware,
        static_origins=static_origins,
        expose_headers=["X-Request-ID"],
    )

    # Tenant context extraction
    app.add_middleware(TenantContextMiddleware)

    # Request ID injection — innermost, runs first
    app.add_middleware(RequestIDMiddleware)

