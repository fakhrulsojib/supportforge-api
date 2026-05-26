"""Unit tests for widget session token management.

Covers:
    - create_widget_token: round-trip with verify, custom expiry,
      empty tenant_id raises ValueError, empty tenant_slug raises ValueError,
      visitor_id included in token
    - verify_widget_token: valid token, expired token raises AuthError,
      tampered token raises AuthError, non-widget token_type raises AuthError,
      empty token raises AuthError, missing tenant_id raises AuthError,
      missing tenant_slug raises AuthError, with/without ws_ prefix,
      wrong signing key raises AuthError
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from jose import jwt

from app.core.exceptions import AuthError
from app.core.widget_token import (
    WIDGET_TOKEN_PREFIX,
    WidgetTokenPayload,
    create_widget_token,
    verify_widget_token,
)

# ── Fixtures ──────────────────────────────────────────────────────

SECRET_KEY = "test-widget-secret-key-for-unit-tests"  # noqa: S105
ALGORITHM = "HS256"


# ── Token Creation Tests ─────────────────────────────────────────


class TestCreateWidgetToken:
    """Tests for widget token creation."""

    def test_round_trip_create_and_verify(self) -> None:
        """Created token should verify successfully and return correct payload."""
        token = create_widget_token(
            tenant_id="tenant-abc",
            tenant_slug="acme",
            secret_key=SECRET_KEY,
        )
        payload = verify_widget_token(token, SECRET_KEY)
        assert isinstance(payload, WidgetTokenPayload)
        assert payload.tenant_id == "tenant-abc"
        assert payload.tenant_slug == "acme"
        assert payload.token_type == "widget"

    def test_token_has_ws_prefix(self) -> None:
        """Widget token string should start with ws_ prefix."""
        token = create_widget_token(
            tenant_id="tenant-1",
            tenant_slug="slug-1",
            secret_key=SECRET_KEY,
        )
        assert token.startswith(WIDGET_TOKEN_PREFIX)

    def test_custom_expiry(self) -> None:
        """Widget token should respect a custom expiry delta."""
        token = create_widget_token(
            tenant_id="t1",
            tenant_slug="s1",
            secret_key=SECRET_KEY,
            expires_delta=timedelta(minutes=30),
        )
        raw = token[len(WIDGET_TOKEN_PREFIX):]
        decoded = jwt.decode(raw, SECRET_KEY, algorithms=[ALGORITHM])
        exp = datetime.fromtimestamp(decoded["exp"], tz=timezone.utc)
        iat = datetime.fromtimestamp(decoded["iat"], tz=timezone.utc)
        delta = exp - iat
        assert abs(delta.total_seconds() - 1800) < 5  # 30min ± 5s

    def test_default_expiry_one_hour(self) -> None:
        """Widget token should default to 1-hour expiry."""
        token = create_widget_token(
            tenant_id="t1",
            tenant_slug="s1",
            secret_key=SECRET_KEY,
        )
        raw = token[len(WIDGET_TOKEN_PREFIX):]
        decoded = jwt.decode(raw, SECRET_KEY, algorithms=[ALGORITHM])
        exp = datetime.fromtimestamp(decoded["exp"], tz=timezone.utc)
        iat = datetime.fromtimestamp(decoded["iat"], tz=timezone.utc)
        delta = exp - iat
        assert abs(delta.total_seconds() - 3600) < 5  # 1h ± 5s

    def test_empty_tenant_id_raises(self) -> None:
        """Empty tenant_id should raise ValueError."""
        with pytest.raises(ValueError, match="tenant_id is required"):
            create_widget_token(
                tenant_id="",
                tenant_slug="slug",
                secret_key=SECRET_KEY,
            )

    def test_empty_tenant_slug_raises(self) -> None:
        """Empty tenant_slug should raise ValueError."""
        with pytest.raises(ValueError, match="tenant_slug is required"):
            create_widget_token(
                tenant_id="tenant-1",
                tenant_slug="",
                secret_key=SECRET_KEY,
            )

    def test_visitor_id_included_in_token(self) -> None:
        """Visitor ID should be encoded into the token payload."""
        token = create_widget_token(
            tenant_id="t1",
            tenant_slug="s1",
            secret_key=SECRET_KEY,
            visitor_id="visitor-xyz-123",
        )
        payload = verify_widget_token(token, SECRET_KEY)
        assert payload.visitor_id == "visitor-xyz-123"

    def test_visitor_id_defaults_to_empty(self) -> None:
        """Visitor ID should default to empty string when not provided."""
        token = create_widget_token(
            tenant_id="t1",
            tenant_slug="s1",
            secret_key=SECRET_KEY,
        )
        payload = verify_widget_token(token, SECRET_KEY)
        assert payload.visitor_id == ""


# ── Token Verification Tests ─────────────────────────────────────


class TestVerifyWidgetToken:
    """Tests for widget token verification."""

    def test_verify_valid_token(self) -> None:
        """Valid widget token should decode successfully."""
        token = create_widget_token(
            tenant_id="tenant-1",
            tenant_slug="acme",
            secret_key=SECRET_KEY,
            visitor_id="v-1",
        )
        payload = verify_widget_token(token, SECRET_KEY)
        assert payload.tenant_id == "tenant-1"
        assert payload.tenant_slug == "acme"
        assert payload.visitor_id == "v-1"
        assert payload.token_type == "widget"
        assert isinstance(payload.exp, datetime)
        assert payload.exp > datetime.now(timezone.utc)

    def test_expired_token_raises(self) -> None:
        """Expired widget token should raise AuthError."""
        token = create_widget_token(
            tenant_id="t1",
            tenant_slug="s1",
            secret_key=SECRET_KEY,
            expires_delta=timedelta(seconds=-10),  # Already expired
        )
        with pytest.raises(AuthError, match="Invalid or expired widget token"):
            verify_widget_token(token, SECRET_KEY)

    def test_tampered_token_raises(self) -> None:
        """Tampered token payload should raise AuthError."""
        token = create_widget_token(
            tenant_id="t1",
            tenant_slug="s1",
            secret_key=SECRET_KEY,
        )
        # Flip a character in the JWT body (after ws_ prefix)
        raw = token[len(WIDGET_TOKEN_PREFIX):]
        parts = raw.split(".")
        # Tamper with the payload portion
        tampered_payload = parts[1][:-1] + ("A" if parts[1][-1] != "A" else "B")
        tampered_raw = f"{parts[0]}.{tampered_payload}.{parts[2]}"
        tampered_token = f"{WIDGET_TOKEN_PREFIX}{tampered_raw}"
        with pytest.raises(AuthError, match="Invalid or expired widget token"):
            verify_widget_token(tampered_token, SECRET_KEY)

    def test_non_widget_token_type_raises(self) -> None:
        """Token with token_type != 'widget' should raise AuthError."""
        payload = {
            "tenant_id": "t1",
            "tenant_slug": "s1",
            "visitor_id": "",
            "token_type": "access",  # Wrong type
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        }
        raw = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        with pytest.raises(AuthError, match="Expected widget token, got access"):
            verify_widget_token(raw, SECRET_KEY)

    def test_empty_token_raises(self) -> None:
        """Empty string should raise AuthError."""
        with pytest.raises(AuthError, match="Empty widget token"):
            verify_widget_token("", SECRET_KEY)

    def test_only_prefix_raises(self) -> None:
        """Token that is just the ws_ prefix should raise AuthError."""
        with pytest.raises(AuthError, match="Empty widget token"):
            verify_widget_token(WIDGET_TOKEN_PREFIX, SECRET_KEY)

    def test_missing_tenant_id_raises(self) -> None:
        """Token without tenant_id should raise AuthError."""
        payload = {
            "tenant_slug": "s1",
            "token_type": "widget",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        }
        raw = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        with pytest.raises(AuthError, match="Widget token missing tenant identifier"):
            verify_widget_token(raw, SECRET_KEY)

    def test_missing_tenant_slug_raises(self) -> None:
        """Token without tenant_slug should raise AuthError."""
        payload = {
            "tenant_id": "t1",
            "token_type": "widget",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        }
        raw = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        with pytest.raises(AuthError, match="Widget token missing tenant slug"):
            verify_widget_token(raw, SECRET_KEY)

    def test_verify_with_ws_prefix(self) -> None:
        """Token with ws_ prefix should verify correctly."""
        token = create_widget_token(
            tenant_id="t1",
            tenant_slug="s1",
            secret_key=SECRET_KEY,
        )
        assert token.startswith(WIDGET_TOKEN_PREFIX)
        payload = verify_widget_token(token, SECRET_KEY)
        assert payload.tenant_id == "t1"

    def test_verify_without_ws_prefix(self) -> None:
        """Token without ws_ prefix should also verify correctly."""
        token = create_widget_token(
            tenant_id="t1",
            tenant_slug="s1",
            secret_key=SECRET_KEY,
        )
        raw = token[len(WIDGET_TOKEN_PREFIX):]
        payload = verify_widget_token(raw, SECRET_KEY)
        assert payload.tenant_id == "t1"

    def test_wrong_signing_key_raises(self) -> None:
        """Token verified with wrong secret should raise AuthError."""
        token = create_widget_token(
            tenant_id="t1",
            tenant_slug="s1",
            secret_key=SECRET_KEY,
        )
        with pytest.raises(AuthError, match="Invalid or expired widget token"):
            verify_widget_token(token, "completely-wrong-secret-key")
