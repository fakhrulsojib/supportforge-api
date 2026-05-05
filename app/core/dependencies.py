"""FastAPI dependency injection factories.

All dependencies are async generators or callables that can be used
with ``fastapi.Depends``. Infrastructure dependencies (DB, LLM, etc.)
are stubbed and will be fully wired in subsequent sub-phases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import Depends, Header

from app.config import Settings, get_settings
from app.core.exceptions import AuthError, TenantNotFoundError
from app.core.security import verify_token
from app.domain.models.enums import UserRole  # noqa: TCH001 — used at runtime in require_role()
from app.domain.models.user import User  # noqa: TCH001 — used at runtime in require_role()
from app.infrastructure.database.connection import get_async_session
from app.infrastructure.database.repositories.user_repo import SQLUserRepository

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession
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


async def get_current_user(
    authorization: str = Header(..., alias="Authorization"),
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_app_settings),
) -> User:
    """Extract and validate the current user from JWT.

    Expects ``Authorization: Bearer <token>`` header.

    Args:
        authorization: Raw Authorization header value.
        session: Database session.
        settings: Application settings.

    Returns:
        The authenticated User domain model.

    Raises:
        AuthError: If token is missing, invalid, or user not found.
    """
    if not authorization.startswith("Bearer "):
        raise AuthError("Authorization header must be 'Bearer <token>'")

    token = authorization[7:]  # Strip "Bearer "
    if not token:
        raise AuthError("Missing access token")

    payload = verify_token(
        token=token,
        secret_key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
        expected_type="access",
    )

    user_repo = SQLUserRepository(session)
    user = await user_repo.get_by_id(payload.user_id)
    if not user:
        raise AuthError("User not found")

    return user


def require_role(*allowed_roles: UserRole) -> Callable[..., User]:
    """Create a dependency that restricts access to specific roles.

    Usage::

        @router.post("/admin-only")
        async def admin_endpoint(
            user: User = Depends(require_role(UserRole.ADMIN)),
        ) -> dict:
            ...

    Args:
        allowed_roles: Roles that are permitted to access the endpoint.

    Returns:
        A FastAPI dependency function.
    """

    async def _check_role(
        user: User = Depends(get_current_user),
    ) -> User:
        if user.role not in allowed_roles:
            allowed = ", ".join(r.value for r in allowed_roles)
            raise AuthError(f"Insufficient permissions. Required role: {allowed}")
        return user

    return _check_role


async def get_db() -> Any:
    """Yield an async database session.

    Placeholder — fully implemented in sub-phase 1.2 (PostgreSQL + Alembic).
    """
    raise NotImplementedError("Database session not configured yet — see sub-phase 1.2")


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


def get_cache(request: Request) -> Any:
    """Return the cache adapter from application state.

    Returns the RedisAdapter initialized during lifespan startup,
    or None if Redis is unavailable (graceful degradation).

    Args:
        request: The current request (used to access app.state).

    Returns:
        CachePort implementation or None.
    """
    return getattr(request.app.state, "cache", None)

