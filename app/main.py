"""FastAPI application factory.

The ``create_app()`` function builds and configures the FastAPI application
with all middleware, exception handlers, routers, and lifecycle events.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.core.events import lifespan
from app.core.exceptions import SupportForgeError
from app.core.middleware import setup_middleware

__version__ = "0.1.0"


def create_app() -> FastAPI:
    """Build the FastAPI application with all configuration applied.

    Returns a fully configured FastAPI instance ready for ``uvicorn``.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Production-grade, multi-tenant AI customer support agent",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────
    setup_middleware(app)

    # ── Exception Handlers ───────────────────────────────────────
    _register_exception_handlers(app)

    # ── Routes ───────────────────────────────────────────────────
    _register_routes(app)

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers for consistent error responses."""

    @app.exception_handler(SupportForgeError)
    async def supportforge_error_handler(request: Request, exc: SupportForgeError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                },
            },
        )


def _register_routes(app: FastAPI) -> None:
    """Register all route handlers on the application."""

    @app.get("/health", tags=["System"])
    async def health_check() -> dict[str, str]:
        """Health check endpoint.

        Returns the application status and version.
        Used by Docker health checks and load balancers.
        """
        return {
            "status": "healthy",
            "version": __version__,
        }

    # API v1 routers
    from app.api.v1.auth import router as auth_router
    from app.api.v1.chat_router import router as chat_router
    from app.api.v1.chat_ws import router as chat_ws_router
    from app.api.v1.conversations import router as conversations_router
    from app.api.v1.ingest import router as ingest_router
    from app.api.v1.review import router as review_router
    from app.api.v1.tenants import router as tenants_router

    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(chat_ws_router)
    app.include_router(conversations_router)
    app.include_router(ingest_router)
    app.include_router(review_router)
    app.include_router(tenants_router)


# Module-level app instance for uvicorn: `uvicorn app.main:app`
app = create_app()
