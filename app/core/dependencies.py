"""FastAPI dependency injection factories.

All dependencies are async generators or callables that can be used
with ``fastapi.Depends``. Infrastructure dependencies (DB, cache,
ChatService, ConnectionManager) are initialized during the lifespan
and injected via ``app.state``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.requests import Request  # noqa: TC002 — FastAPI DI needs this at runtime

from app.config import Settings, get_settings
from app.core.exceptions import AuthError, TenantNotFoundError
from app.core.security import verify_token
from app.domain.models.enums import UserRole  # noqa: TCH001 — used at runtime in require_role()
from app.domain.models.user import User  # noqa: TCH001 — used at runtime in require_role()
from app.infrastructure.database.connection import get_async_session
from app.infrastructure.database.repositories.user_repo import SQLUserRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# ── Security scheme ──────────────────────────────────────────────
# HTTPBearer provides:
#   1. Proper 403 (instead of 422) when Authorization header is missing
#   2. "Authorize" button in Swagger /docs for authenticated endpoints
_bearer_scheme = HTTPBearer()


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
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_app_settings),
) -> User:
    """Extract and validate the current user from JWT.

    Uses ``HTTPBearer`` security scheme which auto-rejects requests
    without a valid ``Authorization: Bearer <token>`` header with
    a 403 response and renders the "Authorize" button in Swagger.

    Args:
        credentials: Parsed Bearer credentials from HTTPBearer.
        session: Database session.
        settings: Application settings.

    Returns:
        The authenticated User domain model.

    Raises:
        AuthError: If token is invalid or user not found.
    """
    token = credentials.credentials
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


def require_role(*allowed_roles: UserRole) -> Any:
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


def get_chat_service(request: Request) -> Any:
    """Return the ChatService singleton from application state.

    Initialized during lifespan startup with LLM, VectorStore,
    and EmbeddingService dependencies.

    Args:
        request: The current request (used to access app.state).

    Returns:
        ChatService instance.

    Raises:
        RuntimeError: If ChatService was not initialized.
    """
    chat_service = getattr(request.app.state, "chat_service", None)
    if chat_service is None:
        msg = "ChatService not initialized — check lifespan startup"
        raise RuntimeError(msg)
    return chat_service


def get_ws_manager(request: Request) -> Any:
    """Return the WebSocket ConnectionManager from application state.

    Initialized during lifespan startup.

    Args:
        request: The current request (used to access app.state).

    Returns:
        ConnectionManager instance.

    Raises:
        RuntimeError: If ConnectionManager was not initialized.
    """
    ws_manager = getattr(request.app.state, "ws_manager", None)
    if ws_manager is None:
        msg = "ConnectionManager not initialized — check lifespan startup"
        raise RuntimeError(msg)
    return ws_manager


def get_embedding_service(request: Request) -> Any:
    """Return the EmbeddingService from application state.

    Initialized during lifespan startup for embedding generation.

    Args:
        request: The current request (used to access app.state).

    Returns:
        EmbeddingService instance.

    Raises:
        RuntimeError: If EmbeddingService was not initialized.
    """
    embedding_service = getattr(request.app.state, "embedding_service", None)
    if embedding_service is None:
        msg = "EmbeddingService not initialized — check lifespan startup"
        raise RuntimeError(msg)
    return embedding_service


def get_vector_store(request: Request) -> Any:
    """Return the VectorStore from application state.

    Initialized during lifespan startup for vector database operations.

    Args:
        request: The current request (used to access app.state).

    Returns:
        VectorStore instance.

    Raises:
        RuntimeError: If VectorStore was not initialized.
    """
    vector_store = getattr(request.app.state, "vector_store", None)
    if vector_store is None:
        msg = "VectorStore not initialized — check lifespan startup"
        raise RuntimeError(msg)
    return vector_store
