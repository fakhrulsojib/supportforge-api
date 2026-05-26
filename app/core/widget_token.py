"""Widget session token management for anonymous SDK visitors.

Widget tokens are short-lived JWTs with a ``ws_`` prefix that allow
anonymous visitors on tenant websites to connect via WebSocket without
a database user row.  They are fundamentally different from user JWTs
(no ``user_id``, no ``role``), so they live in their own module to
preserve type safety in the existing auth pipeline.

Token lifecycle:
    1. Visitor lands on tenant website → SDK calls ``POST /widget/session``
    2. Backend validates embed key → issues ``ws_<jwt>`` token
    3. SDK opens WebSocket with ``?token=ws_<jwt>``
    4. WebSocket handler detects ``ws_`` prefix → uses this module to verify
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import structlog
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from app.core.exceptions import AuthError

logger = structlog.get_logger(__name__)


class WidgetTokenPayload(BaseModel):
    """Decoded widget session token payload.

    Unlike ``TokenPayload``, this has no ``user_id`` or ``role`` —
    widget visitors are anonymous.
    """

    tenant_id: str = Field(description="Tenant the session belongs to")
    tenant_slug: str = Field(description="Tenant slug for display/routing")
    visitor_id: str = Field(default="", description="Optional visitor identifier")
    token_type: str = Field(default="widget", description="Always 'widget'")
    exp: datetime = Field(description="Token expiration timestamp")


# Prefix used to distinguish widget tokens from user JWTs in the WS handler
WIDGET_TOKEN_PREFIX = "ws_"


def create_widget_token(
    *,
    tenant_id: str,
    tenant_slug: str,
    secret_key: str,
    algorithm: str = "HS256",
    visitor_id: str = "",
    expires_delta: timedelta | None = None,
) -> str:
    """Create a short-lived widget session token.

    Args:
        tenant_id: The tenant this session belongs to.
        tenant_slug: Tenant slug for routing.
        secret_key: JWT signing key (same as ``jwt_secret_key``).
        algorithm: JWT algorithm (default HS256).
        visitor_id: Optional visitor identifier for session continuity.
        expires_delta: Token TTL (default 1 hour).

    Returns:
        A ``ws_``-prefixed JWT string.
    """
    if not tenant_id:
        msg = "tenant_id is required for widget token"
        raise ValueError(msg)
    if not tenant_slug:
        msg = "tenant_slug is required for widget token"
        raise ValueError(msg)

    now = datetime.now(timezone.utc)
    if expires_delta is None:
        expires_delta = timedelta(hours=1)

    payload: dict[str, Any] = {
        "tenant_id": tenant_id,
        "tenant_slug": tenant_slug,
        "visitor_id": visitor_id,
        "token_type": "widget",
        "iat": now,
        "exp": now + expires_delta,
    }
    raw = jwt.encode(payload, secret_key, algorithm=algorithm)
    return f"{WIDGET_TOKEN_PREFIX}{raw}"


def verify_widget_token(
    token: str,
    secret_key: str,
    algorithm: str = "HS256",
) -> WidgetTokenPayload:
    """Verify a ``ws_``-prefixed widget session token.

    Args:
        token: The raw token string (with or without ``ws_`` prefix).
        secret_key: JWT signing key for verification.
        algorithm: JWT algorithm.

    Returns:
        Decoded WidgetTokenPayload.

    Raises:
        AuthError: If the token is expired, malformed, or wrong type.
    """
    # Strip the prefix if present
    raw = token[len(WIDGET_TOKEN_PREFIX):] if token.startswith(WIDGET_TOKEN_PREFIX) else token

    if not raw:
        raise AuthError("Empty widget token")

    try:
        payload = jwt.decode(raw, secret_key, algorithms=[algorithm])
    except JWTError as e:
        logger.warning("widget_token_verification_failed", error=str(e))
        raise AuthError("Invalid or expired widget token") from e

    token_type = payload.get("token_type", "")
    if token_type != "widget":
        raise AuthError(f"Expected widget token, got {token_type}")

    tenant_id = payload.get("tenant_id", "")
    if not tenant_id:
        raise AuthError("Widget token missing tenant identifier")

    tenant_slug = payload.get("tenant_slug", "")
    if not tenant_slug:
        raise AuthError("Widget token missing tenant slug")

    return WidgetTokenPayload(
        tenant_id=tenant_id,
        tenant_slug=tenant_slug,
        visitor_id=payload.get("visitor_id", ""),
        token_type=token_type,
        exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
    )
