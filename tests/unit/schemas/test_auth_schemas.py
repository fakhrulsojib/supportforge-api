"""Schema validation tests for authentication DTOs.

Tests edge cases for Pydantic field constraints:
  - min/max length enforcement
  - required fields
  - default values
  - boundary values
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas.auth import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
)


class TestRegisterRequest:
    """Edge-case validation for RegisterRequest."""

    def test_valid_request(self) -> None:
        req = RegisterRequest(
            email="user@example.com",
            password="StrongP@ss1",
            tenant_id="t-1",
        )
        assert req.role == "viewer"  # default

    def test_empty_email_rejected(self) -> None:
        with pytest.raises(ValidationError, match="email"):
            RegisterRequest(email="", password="StrongP@ss1", tenant_id="t-1")

    def test_email_max_length(self) -> None:
        long_email = "a" * 321
        with pytest.raises(ValidationError, match="email"):
            RegisterRequest(email=long_email, password="StrongP@ss1", tenant_id="t-1")

    def test_password_too_short(self) -> None:
        with pytest.raises(ValidationError, match="password"):
            RegisterRequest(email="a@b.c", password="Short1!", tenant_id="t-1")

    def test_password_max_length(self) -> None:
        long_pw = "A" * 129
        with pytest.raises(ValidationError, match="password"):
            RegisterRequest(email="a@b.c", password=long_pw, tenant_id="t-1")

    def test_password_at_min_boundary(self) -> None:
        req = RegisterRequest(email="a@b.c", password="Abcdef1!", tenant_id="t-1")
        assert len(req.password) == 8

    def test_empty_tenant_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="tenant_id"):
            RegisterRequest(email="a@b.c", password="StrongP@ss1", tenant_id="")

    def test_missing_email_rejected(self) -> None:
        with pytest.raises(ValidationError, match="email"):
            RegisterRequest(password="StrongP@ss1", tenant_id="t-1")  # type: ignore[call-arg]

    def test_custom_role(self) -> None:
        req = RegisterRequest(
            email="admin@test.com",
            password="StrongP@ss1",
            tenant_id="t-1",
            role="admin",
        )
        assert req.role == "admin"


class TestLoginRequest:
    """Edge-case validation for LoginRequest."""

    def test_valid_request(self) -> None:
        req = LoginRequest(email="user@test.com", password="StrongP@ss1", tenant_id="t-1")
        assert req.email == "user@test.com"

    def test_empty_email_rejected(self) -> None:
        with pytest.raises(ValidationError, match="email"):
            LoginRequest(email="", password="StrongP@ss1", tenant_id="t-1")

    def test_short_password_rejected(self) -> None:
        """M7: Passwords under 8 chars should be rejected at schema level."""
        with pytest.raises(ValidationError, match="password"):
            LoginRequest(email="a@b.c", password="Short1!", tenant_id="t-1")

    def test_empty_tenant_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="tenant_id"):
            LoginRequest(email="a@b.c", password="StrongP@ss1", tenant_id="")


class TestRefreshRequest:
    """Edge-case validation for RefreshRequest."""

    def test_valid_request(self) -> None:
        req = RefreshRequest(refresh_token="some-token")
        assert req.refresh_token == "some-token"

    def test_empty_token_rejected(self) -> None:
        with pytest.raises(ValidationError, match="refresh_token"):
            RefreshRequest(refresh_token="")

    def test_missing_token_rejected(self) -> None:
        with pytest.raises(ValidationError, match="refresh_token"):
            RefreshRequest()  # type: ignore[call-arg]


class TestTokenResponse:
    """Edge-case validation for TokenResponse."""

    def test_valid_response(self) -> None:
        resp = TokenResponse(
            access_token="at",
            refresh_token="rt",
            expires_in=900,
        )
        assert resp.token_type == "bearer"  # default

    def test_custom_token_type(self) -> None:
        resp = TokenResponse(
            access_token="at",
            refresh_token="rt",
            token_type="mac",
            expires_in=900,
        )
        assert resp.token_type == "mac"

    def test_missing_access_token_rejected(self) -> None:
        with pytest.raises(ValidationError, match="access_token"):
            TokenResponse(refresh_token="rt", expires_in=900)  # type: ignore[call-arg]

    def test_missing_expires_in_rejected(self) -> None:
        with pytest.raises(ValidationError, match="expires_in"):
            TokenResponse(access_token="at", refresh_token="rt")  # type: ignore[call-arg]
