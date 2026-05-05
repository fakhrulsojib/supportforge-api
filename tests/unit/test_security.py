"""Unit tests for JWT security module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from jose import jwt

from app.core.exceptions import AuthError
from app.core.security import (
    TokenPayload,
    create_access_token,
    create_refresh_token,
    hash_password,
    validate_password_strength,
    verify_password,
    verify_token,
)

# ── Fixtures ──────────────────────────────────────────────────────

SECRET_KEY = "test-secret-key-for-unit-tests"  # noqa: S105
ALGORITHM = "HS256"


# ── Password Hashing Tests ───────────────────────────────────────


class TestPasswordHashing:
    """Tests for bcrypt password hashing and verification."""

    def test_hash_password_returns_bcrypt_hash(self) -> None:
        """Hashed password should start with $2b$ (bcrypt marker)."""
        hashed = hash_password("TestPass1!")
        assert hashed.startswith("$2b$")

    def test_hash_password_different_each_time(self) -> None:
        """Bcrypt uses random salt — same password should produce different hashes."""
        h1 = hash_password("TestPass1!")
        h2 = hash_password("TestPass1!")
        assert h1 != h2

    def test_verify_password_correct(self) -> None:
        """Correct password should verify against its hash."""
        password = "MyStr0ng!Pass"
        hashed = hash_password(password)
        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self) -> None:
        """Wrong password should not verify."""
        hashed = hash_password("CorrectPass1!")
        assert verify_password("WrongPass1!", hashed) is False

    def test_verify_password_empty_string(self) -> None:
        """Empty string should not match a real hash."""
        hashed = hash_password("RealPass1!")
        assert verify_password("", hashed) is False


# ── Password Validation Tests ────────────────────────────────────


class TestPasswordValidation:
    """Tests for password strength validation."""

    def test_valid_password(self) -> None:
        """A password meeting all requirements should pass."""
        errors = validate_password_strength("StrongP@ss1")
        assert errors == []

    def test_too_short(self) -> None:
        """Password under 8 chars should fail."""
        errors = validate_password_strength("Ab1!")
        assert any("at least 8" in e for e in errors)

    def test_too_long(self) -> None:
        """Password over 128 chars should fail."""
        errors = validate_password_strength("A" * 129 + "a1!")
        assert any("at most 128" in e for e in errors)

    def test_missing_uppercase(self) -> None:
        """Password without uppercase should fail."""
        errors = validate_password_strength("lowercase1!")
        assert any("uppercase" in e for e in errors)

    def test_missing_lowercase(self) -> None:
        """Password without lowercase should fail."""
        errors = validate_password_strength("UPPERCASE1!")
        assert any("lowercase" in e for e in errors)

    def test_missing_digit(self) -> None:
        """Password without digit should fail."""
        errors = validate_password_strength("NoDigits!!")
        assert any("digit" in e for e in errors)

    def test_missing_special(self) -> None:
        """Password without special char should fail."""
        errors = validate_password_strength("NoSpecial1A")
        assert any("special" in e for e in errors)

    def test_multiple_failures(self) -> None:
        """Password failing multiple rules should report all errors."""
        errors = validate_password_strength("short")
        assert len(errors) >= 3  # too short + missing uppercase + missing digit + missing special


# ── Token Creation Tests ─────────────────────────────────────────


class TestTokenCreation:
    """Tests for JWT token creation."""

    def test_create_access_token_returns_string(self) -> None:
        """Access token should be a non-empty string."""
        token = create_access_token(
            user_id="user-1",
            tenant_id="tenant-1",
            role="admin",
            secret_key=SECRET_KEY,
        )
        assert isinstance(token, str)
        assert len(token) > 0

    def test_access_token_payload_contents(self) -> None:
        """Access token payload should contain expected claims."""
        token = create_access_token(
            user_id="user-123",
            tenant_id="tenant-456",
            role="agent",
            secret_key=SECRET_KEY,
            algorithm=ALGORITHM,
        )
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "user-123"
        assert payload["tenant_id"] == "tenant-456"
        assert payload["role"] == "agent"
        assert payload["token_type"] == "access"
        assert "exp" in payload
        assert "iat" in payload

    def test_access_token_default_expiry(self) -> None:
        """Access token should default to 15min expiry."""
        token = create_access_token(
            user_id="u1",
            tenant_id="t1",
            role="viewer",
            secret_key=SECRET_KEY,
        )
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        iat = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
        delta = exp - iat
        assert abs(delta.total_seconds() - 900) < 5  # 15min ± 5s

    def test_access_token_custom_expiry(self) -> None:
        """Access token should respect custom expiry."""
        token = create_access_token(
            user_id="u1",
            tenant_id="t1",
            role="viewer",
            secret_key=SECRET_KEY,
            expires_minutes=30,
        )
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        iat = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
        delta = exp - iat
        assert abs(delta.total_seconds() - 1800) < 5  # 30min ± 5s

    def test_create_refresh_token_returns_string(self) -> None:
        """Refresh token should be a non-empty string."""
        token = create_refresh_token(
            user_id="user-1",
            tenant_id="tenant-1",
            secret_key=SECRET_KEY,
        )
        assert isinstance(token, str)
        assert len(token) > 0

    def test_refresh_token_payload(self) -> None:
        """Refresh token should have token_type='refresh'."""
        token = create_refresh_token(
            user_id="user-1",
            tenant_id="tenant-1",
            secret_key=SECRET_KEY,
        )
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["token_type"] == "refresh"
        assert payload["sub"] == "user-1"

    def test_refresh_token_7d_expiry(self) -> None:
        """Refresh token should default to 7-day expiry."""
        token = create_refresh_token(
            user_id="u1",
            tenant_id="t1",
            secret_key=SECRET_KEY,
        )
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        iat = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
        delta = exp - iat
        assert abs(delta.total_seconds() - 604800) < 5  # 7d ± 5s


# ── Token Verification Tests ─────────────────────────────────────


class TestTokenVerification:
    """Tests for JWT token verification."""

    def test_verify_valid_access_token(self) -> None:
        """Valid access token should decode successfully."""
        token = create_access_token(
            user_id="user-1",
            tenant_id="tenant-1",
            role="admin",
            secret_key=SECRET_KEY,
        )
        payload = verify_token(token, SECRET_KEY, expected_type="access")
        assert isinstance(payload, TokenPayload)
        assert payload.user_id == "user-1"
        assert payload.tenant_id == "tenant-1"
        assert payload.role == "admin"
        assert payload.token_type == "access"

    def test_verify_valid_refresh_token(self) -> None:
        """Valid refresh token should decode when expected_type='refresh'."""
        token = create_refresh_token(
            user_id="user-1",
            tenant_id="tenant-1",
            secret_key=SECRET_KEY,
        )
        payload = verify_token(token, SECRET_KEY, expected_type="refresh")
        assert payload.token_type == "refresh"

    def test_verify_expired_token_raises(self) -> None:
        """Expired token should raise AuthError."""
        token = create_access_token(
            user_id="user-1",
            tenant_id="tenant-1",
            role="admin",
            secret_key=SECRET_KEY,
            expires_minutes=-1,  # Already expired
        )
        with pytest.raises(AuthError, match="Invalid or expired token"):
            verify_token(token, SECRET_KEY)

    def test_verify_wrong_secret_raises(self) -> None:
        """Token verified with wrong secret should raise AuthError."""
        token = create_access_token(
            user_id="user-1",
            tenant_id="tenant-1",
            role="admin",
            secret_key=SECRET_KEY,
        )
        with pytest.raises(AuthError, match="Invalid or expired token"):
            verify_token(token, "wrong-secret-key")

    def test_verify_malformed_token_raises(self) -> None:
        """Malformed token string should raise AuthError."""
        with pytest.raises(AuthError, match="Invalid or expired token"):
            verify_token("not.a.valid.jwt.token", SECRET_KEY)

    def test_verify_empty_token_raises(self) -> None:
        """Empty string should raise AuthError."""
        with pytest.raises(AuthError, match="Invalid or expired token"):
            verify_token("", SECRET_KEY)

    def test_verify_wrong_type_raises(self) -> None:
        """Refresh token verified as access should raise AuthError."""
        token = create_refresh_token(
            user_id="user-1",
            tenant_id="tenant-1",
            secret_key=SECRET_KEY,
        )
        with pytest.raises(AuthError, match="Expected access token"):
            verify_token(token, SECRET_KEY, expected_type="access")

    def test_verify_access_as_refresh_raises(self) -> None:
        """Access token verified as refresh should raise AuthError."""
        token = create_access_token(
            user_id="user-1",
            tenant_id="tenant-1",
            role="admin",
            secret_key=SECRET_KEY,
        )
        with pytest.raises(AuthError, match="Expected refresh token"):
            verify_token(token, SECRET_KEY, expected_type="refresh")

    def test_verify_token_missing_sub_raises(self) -> None:
        """Token without 'sub' claim should raise AuthError."""
        payload = {
            "tenant_id": "t1",
            "role": "admin",
            "token_type": "access",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=15),
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        with pytest.raises(AuthError, match="Token missing user identifier"):
            verify_token(token, SECRET_KEY)

    def test_verify_token_exp_field(self) -> None:
        """Verified token should have exp datetime populated."""
        token = create_access_token(
            user_id="user-1",
            tenant_id="tenant-1",
            role="admin",
            secret_key=SECRET_KEY,
        )
        payload = verify_token(token, SECRET_KEY)
        assert payload.exp is not None
        assert isinstance(payload.exp, datetime)
        assert payload.exp > datetime.now(timezone.utc)
