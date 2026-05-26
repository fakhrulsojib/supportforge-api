"""Middleware for CORS, request-ID injection, and tenant context extraction."""

from __future__ import annotations

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


def setup_middleware(app: FastAPI) -> None:
    """Register all middleware on the FastAPI application.

    Order matters — middleware is executed in reverse registration order.
    We register TenantContext first (executes last), then RequestID
    (executes first), so request-ID is available for tenant logging.
    """
    from fastapi.middleware.cors import CORSMiddleware

    from app.config import get_settings

    settings = get_settings()

    # CORS — outermost middleware
    all_origins = settings.cors_origin_list + settings.widget_cors_origin_list
    # CORS spec: allow_credentials=True is incompatible with allow_origins=["*"]
    use_credentials = "*" not in all_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=all_origins,
        allow_credentials=use_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # Tenant context extraction
    app.add_middleware(TenantContextMiddleware)

    # Request ID injection — innermost, runs first
    app.add_middleware(RequestIDMiddleware)
