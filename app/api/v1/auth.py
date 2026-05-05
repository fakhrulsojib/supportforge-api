"""Authentication API router — register, login, refresh.

All endpoints operate within a tenant context. Users are scoped
to a specific tenant, and email uniqueness is enforced per-tenant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends

from app.api.schemas.auth import LoginRequest, RefreshRequest, RegisterRequest, TokenResponse
from app.core.dependencies import get_app_settings
from app.core.exceptions import AuthError, SupportForgeError
from app.core.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    validate_password_strength,
    verify_password,
    verify_token,
)
from app.domain.models.enums import UserRole
from app.infrastructure.database.connection import get_async_session
from app.infrastructure.database.repositories.tenant_repo import SQLTenantRepository
from app.infrastructure.database.repositories.user_repo import SQLUserRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.config import Settings

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(
    request: RegisterRequest,
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_app_settings),
) -> TokenResponse:
    """Register a new user and return JWT tokens.

    Validates:
        - Tenant exists
        - Email not already taken within tenant
        - Password meets strength requirements
        - Role is valid

    Args:
        request: Registration data (email, password, tenant_id, role).
        session: Database session.
        settings: Application settings.

    Returns:
        TokenResponse with access and refresh tokens.

    Raises:
        SupportForgeError(404): Tenant not found.
        SupportForgeError(409): Email already registered.
        SupportForgeError(422): Password too weak or invalid role.
    """
    # Validate tenant exists
    tenant_repo = SQLTenantRepository(session)
    tenant = await tenant_repo.get_by_id(request.tenant_id)
    if not tenant:
        raise SupportForgeError(
            message=f"Tenant '{request.tenant_id}' not found",
            status_code=404,
            error_code="TENANT_NOT_FOUND",
        )

    # Validate role
    try:
        role = UserRole(request.role)
    except ValueError as exc:
        valid_roles = ", ".join(r.value for r in UserRole)
        raise SupportForgeError(
            message=f"Invalid role '{request.role}'. Must be one of: {valid_roles}",
            status_code=422,
            error_code="INVALID_ROLE",
        ) from exc

    # Validate password strength
    password_errors = validate_password_strength(request.password)
    if password_errors:
        raise SupportForgeError(
            message="; ".join(password_errors),
            status_code=422,
            error_code="WEAK_PASSWORD",
        )

    # Check email uniqueness within tenant
    user_repo = SQLUserRepository(session)
    existing = await user_repo.get_by_email(request.email, request.tenant_id)
    if existing:
        raise SupportForgeError(
            message=f"Email '{request.email}' is already registered in this tenant",
            status_code=409,
            error_code="EMAIL_ALREADY_EXISTS",
        )

    # Create user with hashed password
    from app.domain.models.user import UserCreate

    user_create = UserCreate(email=request.email, password=request.password, role=role)
    user = await user_repo.create(request.tenant_id, user_create)

    # Update password hash (repo creates with empty hash)
    from app.infrastructure.database.models import UserModel

    user_model = await session.get(UserModel, user.id)
    if user_model:
        user_model.password_hash = hash_password(request.password)
        await session.flush()

    logger.info("user_registered", user_id=user.id, tenant_id=request.tenant_id, role=role.value)

    # Generate tokens
    access_token = create_access_token(
        user_id=user.id,
        tenant_id=request.tenant_id,
        role=role.value,
        secret_key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
        expires_minutes=settings.jwt_access_token_expire_minutes,
    )
    refresh_token = create_refresh_token(
        user_id=user.id,
        tenant_id=request.tenant_id,
        secret_key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
        expires_days=settings.jwt_refresh_token_expire_days,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.jwt_access_token_expire_minutes * 60,
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_app_settings),
) -> TokenResponse:
    """Authenticate a user and return JWT tokens.

    Args:
        request: Login credentials (email, password, tenant_id).
        session: Database session.
        settings: Application settings.

    Returns:
        TokenResponse with access and refresh tokens.

    Raises:
        AuthError: Invalid credentials.
    """
    user_repo = SQLUserRepository(session)
    user = await user_repo.get_by_email(request.email, request.tenant_id)

    if not user:
        raise AuthError("Invalid email or password")

    if not verify_password(request.password, user.password_hash):
        raise AuthError("Invalid email or password")

    logger.info("user_login", user_id=user.id, tenant_id=user.tenant_id)

    access_token = create_access_token(
        user_id=user.id,
        tenant_id=user.tenant_id,
        role=user.role.value,
        secret_key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
        expires_minutes=settings.jwt_access_token_expire_minutes,
    )
    refresh_token = create_refresh_token(
        user_id=user.id,
        tenant_id=user.tenant_id,
        secret_key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
        expires_days=settings.jwt_refresh_token_expire_days,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.jwt_access_token_expire_minutes * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    request: RefreshRequest,
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_app_settings),
) -> TokenResponse:
    """Refresh an access token using a valid refresh token.

    Args:
        request: Refresh token.
        session: Database session.
        settings: Application settings.

    Returns:
        TokenResponse with new access token (same refresh token).

    Raises:
        AuthError: Invalid or expired refresh token.
    """
    payload = verify_token(
        token=request.refresh_token,
        secret_key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
        expected_type="refresh",
    )

    # Verify user still exists
    user_repo = SQLUserRepository(session)
    user = await user_repo.get_by_id(payload.user_id)
    if not user:
        raise AuthError("User no longer exists")

    access_token = create_access_token(
        user_id=user.id,
        tenant_id=user.tenant_id,
        role=user.role.value,
        secret_key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
        expires_minutes=settings.jwt_access_token_expire_minutes,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=request.refresh_token,
        expires_in=settings.jwt_access_token_expire_minutes * 60,
    )
