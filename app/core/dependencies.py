"""FastAPI dependency injection factories.

All dependencies are async generators or callables that can be used
with ``fastapi.Depends``. Infrastructure dependencies (DB, LLM, etc.)
are stubbed and will be fully wired in subsequent sub-phases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.config import Settings, get_settings
from app.core.exceptions import TenantNotFoundError

if TYPE_CHECKING:
    from starlette.requests import Request


def get_app_settings() -> Settings:
    """Return the application settings singleton."""
    return get_settings()


def get_tenant_id(request: Request) -> str:
    """Extract tenant ID from request state (set by TenantContextMiddleware).

    Raises TenantNotFoundError if no X-Tenant-ID header was provided.
    """
    tenant_id: str | None = getattr(request.state, "tenant_id", None)
    if not tenant_id:
        raise TenantNotFoundError("Missing X-Tenant-ID header")
    return tenant_id


async def get_db() -> Any:
    """Yield an async database session.

    Placeholder — fully implemented in sub-phase 1.2 (PostgreSQL + Alembic).
    """
    raise NotImplementedError("Database session not configured yet — see sub-phase 1.2")


async def get_current_user() -> Any:
    """Extract and validate the current user from JWT.

    Placeholder — fully implemented in sub-phase 2.5 (JWT Authentication).
    """
    raise NotImplementedError("JWT auth not configured yet — see sub-phase 2.5")


async def get_llm_provider() -> Any:
    """Return the configured LLM provider adapter.

    Placeholder — fully implemented in sub-phase 1.3 (Ollama Adapter).
    """
    raise NotImplementedError("LLM provider not configured yet — see sub-phase 1.3")


async def get_vector_store() -> Any:
    """Return the configured vector store adapter.

    Placeholder — fully implemented in sub-phase 1.4 (ChromaDB).
    """
    raise NotImplementedError("Vector store not configured yet — see sub-phase 1.4")
