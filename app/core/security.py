"""JWT token management and password hashing.

Provides functions for creating/verifying JWT access and refresh tokens,
as well as bcrypt-based password hashing and verification.

All token operations raise ``AuthError`` on failure for consistent
error handling upstream.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from app.core.exceptions import AuthError

logger = structlog.get_logger(__name__)

# ── Password hashing ─────────────────────────────────────────────

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt.

    Args:
        password: The plaintext password.

    Returns:
        The bcrypt hash string.
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a bcrypt hash.

    Args:
        plain_password: The plaintext password to check.
        hashed_password: The stored bcrypt hash.

    Returns:
        True if the password matches, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)


# ── Password validation ──────────────────────────────────────────

_PASSWORD_MIN_LENGTH = 8
_PASSWORD_MAX_LENGTH = 128

# Compiled patterns for password strength validation
_HAS_UPPERCASE = re.compile(r"[A-Z]")
_HAS_LOWERCASE = re.compile(r"[a-z]")
_HAS_DIGIT = re.compile(r"\d")
_HAS_SPECIAL = re.compile(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?`~]")


def validate_password_strength(password: str) -> list[str]:
    """Validate password meets strength requirements.

    Requirements:
        - 8–128 characters
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one digit
        - At least one special character

    Args:
        password: The password to validate.

    Returns:
        List of validation error messages. Empty list means valid.
    """
    errors: list[str] = []

    if len(password) < _PASSWORD_MIN_LENGTH:
        errors.append(f"Password must be at least {_PASSWORD_MIN_LENGTH} characters")
    if len(password) > _PASSWORD_MAX_LENGTH:
        errors.append(f"Password must be at most {_PASSWORD_MAX_LENGTH} characters")
    if not _HAS_UPPERCASE.search(password):
        errors.append("Password must contain at least one uppercase letter")
    if not _HAS_LOWERCASE.search(password):
        errors.append("Password must contain at least one lowercase letter")
    if not _HAS_DIGIT.search(password):
        errors.append("Password must contain at least one digit")
    if not _HAS_SPECIAL.search(password):
        errors.append("Password must contain at least one special character")

    return errors


# ── Token models ─────────────────────────────────────────────────


class TokenPayload(BaseModel):
    """Decoded JWT token payload."""

    user_id: str
    tenant_id: str
    role: str | None = Field(description="User role. None for refresh tokens.")
    token_type: str = Field(description="'access' or 'refresh'")
    exp: datetime = Field(description="Token expiration timestamp")


# ── Token creation ───────────────────────────────────────────────


def create_access_token(
    user_id: str,
    tenant_id: str,
    role: str,
    secret_key: str,
    algorithm: str = "HS256",
    expires_minutes: int = 15,
) -> str:
    """Create a signed JWT access token.

    Args:
        user_id: The user's unique identifier.
        tenant_id: The tenant the user belongs to.
        role: User role (admin, agent, viewer).
        secret_key: Signing key.
        algorithm: JWT algorithm (default HS256).
        expires_minutes: Token TTL in minutes.

    Returns:
        Encoded JWT string.
    """
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "role": role,
        "token_type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, secret_key, algorithm=algorithm)


def create_refresh_token(
    user_id: str,
    tenant_id: str,
    secret_key: str,
    algorithm: str = "HS256",
    expires_days: int = 7,
) -> str:
    """Create a signed JWT refresh token.

    Args:
        user_id: The user's unique identifier.
        tenant_id: The tenant the user belongs to.
        secret_key: Signing key.
        algorithm: JWT algorithm.
        expires_days: Token TTL in days.

    Returns:
        Encoded JWT string.
    """
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "token_type": "refresh",
        "iat": now,
        "exp": now + timedelta(days=expires_days),
    }
    return jwt.encode(payload, secret_key, algorithm=algorithm)


# ── Token verification ───────────────────────────────────────────


def verify_token(
    token: str,
    secret_key: str,
    algorithm: str = "HS256",
    expected_type: str = "access",
) -> TokenPayload:
    """Decode and validate a JWT token.

    Args:
        token: The raw JWT string.
        secret_key: Signing key for verification.
        algorithm: JWT algorithm.
        expected_type: Expected token_type ('access' or 'refresh').

    Returns:
        Decoded TokenPayload.

    Raises:
        AuthError: If the token is expired, malformed, or wrong type.
    """
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
    except JWTError as e:
        logger.warning("jwt_verification_failed", error=str(e))
        raise AuthError("Invalid or expired token") from e

    token_type = payload.get("token_type", "")
    if token_type != expected_type:
        raise AuthError(f"Expected {expected_type} token, got {token_type}")

    user_id = payload.get("sub", "")
    if not user_id:
        raise AuthError("Token missing user identifier")

    tenant_id = payload.get("tenant_id", "")
    if not tenant_id:
        raise AuthError("Token missing tenant identifier")

    return TokenPayload(
        user_id=user_id,
        tenant_id=tenant_id,
        role=payload.get("role"),
        token_type=token_type,
        exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
    )
