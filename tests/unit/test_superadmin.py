"""Unit tests for Platform Superadmin role — Phase 9.

Tests cover:
- UserRole enum expansion (SUPERADMIN value)
- User domain model is_superadmin property
- JWT TokenPayload is_superadmin field
- create_access_token / verify_token with is_superadmin claim
- Backward compatibility (tokens without is_superadmin claim)
- require_superadmin() dependency
- require_role() accepting superadmin when ADMIN is allowed
- Superadmin self-registration blocked
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt

from app.config import get_settings
from app.core.exceptions import AuthError
from app.core.security import (
    TokenPayload,
    create_access_token,
    verify_token,
)
from app.domain.models.enums import UserRole
from app.domain.models.user import User

# ── Test JWT secret — must match .env default ──
_JWT_SECRET = "change-me-to-another-random-secret"  # noqa: S105
_ALGORITHM = "HS256"


# ═══════════════════════════════════════════════════════════════════
# Task 1: UserRole enum has SUPERADMIN
# ═══════════════════════════════════════════════════════════════════


class TestUserRoleSuperadmin:
    """Verify SUPERADMIN is a valid UserRole member."""

    def test_superadmin_value_exists(self) -> None:
        """UserRole should have a SUPERADMIN member with value 'superadmin'."""
        assert UserRole.SUPERADMIN.value == "superadmin"

    def test_superadmin_constructible_from_string(self) -> None:
        """UserRole('superadmin') should return SUPERADMIN."""
        role = UserRole("superadmin")
        assert role is UserRole.SUPERADMIN

    def test_superadmin_is_str_enum(self) -> None:
        """SUPERADMIN should be usable as a string."""
        assert str(UserRole.SUPERADMIN) == "UserRole.SUPERADMIN"
        assert UserRole.SUPERADMIN == "superadmin"

    def test_all_roles_present(self) -> None:
        """UserRole should now have 4 members."""
        assert len(UserRole) == 4
        values = {r.value for r in UserRole}
        assert values == {"admin", "agent", "viewer", "superadmin"}


# ═══════════════════════════════════════════════════════════════════
# Task 2: User domain model is_superadmin property
# ═══════════════════════════════════════════════════════════════════


class TestUserIsSuperadmin:
    """Verify is_superadmin property on User domain model."""

    def test_superadmin_role_returns_true(self) -> None:
        """User with SUPERADMIN role should have is_superadmin=True."""
        user = User(id="u1", tenant_id="t1", email="sa@test.com", role=UserRole.SUPERADMIN)
        assert user.is_superadmin is True

    def test_admin_role_returns_false(self) -> None:
        """User with ADMIN role should have is_superadmin=False."""
        user = User(id="u1", tenant_id="t1", email="a@test.com", role=UserRole.ADMIN)
        assert user.is_superadmin is False

    def test_viewer_role_returns_false(self) -> None:
        """User with VIEWER role should have is_superadmin=False."""
        user = User(id="u1", tenant_id="t1", email="v@test.com", role=UserRole.VIEWER)
        assert user.is_superadmin is False

    def test_agent_role_returns_false(self) -> None:
        """User with AGENT role should have is_superadmin=False."""
        user = User(id="u1", tenant_id="t1", email="ag@test.com", role=UserRole.AGENT)
        assert user.is_superadmin is False


# ═══════════════════════════════════════════════════════════════════
# Task 4: JWT TokenPayload + create/verify with is_superadmin
# ═══════════════════════════════════════════════════════════════════


class TestTokenPayloadSuperadmin:
    """Verify is_superadmin field on TokenPayload."""

    def test_default_is_false(self) -> None:
        """TokenPayload.is_superadmin should default to False."""
        tp = TokenPayload(
            user_id="u1",
            tenant_id="t1",
            role="admin",
            token_type="access",
            exp=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
        assert tp.is_superadmin is False

    def test_explicit_true(self) -> None:
        """TokenPayload.is_superadmin can be set to True."""
        tp = TokenPayload(
            user_id="u1",
            tenant_id="t1",
            role="superadmin",
            token_type="access",
            exp=datetime.now(timezone.utc) + timedelta(minutes=15),
            is_superadmin=True,
        )
        assert tp.is_superadmin is True


class TestCreateAccessTokenSuperadmin:
    """Verify create_access_token includes is_superadmin claim."""

    def test_superadmin_claim_included_when_true(self) -> None:
        """JWT payload should contain is_superadmin=True when set."""
        token = create_access_token(
            user_id="u1",
            tenant_id="t1",
            role="superadmin",
            secret_key=_JWT_SECRET,
            is_superadmin=True,
        )
        payload = jwt.decode(token, _JWT_SECRET, algorithms=[_ALGORITHM])
        assert payload["is_superadmin"] is True

    def test_superadmin_claim_absent_when_false(self) -> None:
        """JWT payload should NOT contain is_superadmin when False (default)."""
        token = create_access_token(
            user_id="u1",
            tenant_id="t1",
            role="admin",
            secret_key=_JWT_SECRET,
        )
        payload = jwt.decode(token, _JWT_SECRET, algorithms=[_ALGORITHM])
        assert "is_superadmin" not in payload


class TestVerifyTokenSuperadmin:
    """Verify verify_token reads is_superadmin from payload."""

    def test_superadmin_token_parsed(self) -> None:
        """verify_token should set is_superadmin=True from JWT claim."""
        token = create_access_token(
            user_id="u1",
            tenant_id="t1",
            role="superadmin",
            secret_key=_JWT_SECRET,
            is_superadmin=True,
        )
        payload = verify_token(token, _JWT_SECRET, expected_type="access")
        assert payload.is_superadmin is True
        assert payload.role == "superadmin"

    def test_regular_token_defaults_false(self) -> None:
        """verify_token should set is_superadmin=False for regular tokens."""
        token = create_access_token(
            user_id="u1",
            tenant_id="t1",
            role="admin",
            secret_key=_JWT_SECRET,
        )
        payload = verify_token(token, _JWT_SECRET, expected_type="access")
        assert payload.is_superadmin is False

    def test_backward_compat_old_token_without_claim(self) -> None:
        """Tokens created before Phase 9 (no is_superadmin claim) must still parse."""
        # Simulate a pre-Phase-9 token by manually encoding without is_superadmin
        now = datetime.now(timezone.utc)
        raw_payload = {
            "sub": "u-old",
            "tenant_id": "t-old",
            "role": "admin",
            "token_type": "access",
            "iat": now,
            "exp": now + timedelta(minutes=15),
        }
        token = jwt.encode(raw_payload, _JWT_SECRET, algorithm=_ALGORITHM)
        parsed = verify_token(token, _JWT_SECRET, expected_type="access")
        assert parsed.is_superadmin is False
        assert parsed.user_id == "u-old"


# ═══════════════════════════════════════════════════════════════════
# Task 3: require_superadmin() dependency
# ═══════════════════════════════════════════════════════════════════


class TestRequireSuperadmin:
    """Verify require_superadmin() dependency function."""

    @pytest.mark.asyncio
    async def test_accepts_superadmin_user(self) -> None:
        """Superadmin user should pass require_superadmin check."""
        from app.core.dependencies import require_superadmin

        user = User(
            id="sa-1", tenant_id="t1", email="sa@test.com",
            role=UserRole.SUPERADMIN,
        )
        settings = get_settings()
        token = create_access_token(
            user_id=user.id,
            tenant_id=user.tenant_id,
            role=user.role.value,
            secret_key=settings.jwt_secret_key,
            is_superadmin=True,
        )
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        mock_session = AsyncMock()

        with patch("app.core.dependencies.SQLUserRepository") as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_id = AsyncMock(return_value=user)

            dep = require_superadmin()
            result = await dep(
                user=await self._get_user(credentials, mock_session, settings),
            )
            assert result.role == UserRole.SUPERADMIN

    @pytest.mark.asyncio
    async def test_rejects_admin_user(self) -> None:
        """Admin user should be rejected by require_superadmin with AuthError."""
        from app.core.dependencies import require_superadmin

        user = User(
            id="a-1", tenant_id="t1", email="admin@test.com",
            role=UserRole.ADMIN,
        )

        dep = require_superadmin()
        with pytest.raises(AuthError, match="Superadmin access required"):
            await dep(user=user)

    @pytest.mark.asyncio
    async def test_rejects_viewer_user(self) -> None:
        """Viewer user should be rejected by require_superadmin."""
        from app.core.dependencies import require_superadmin

        user = User(
            id="v-1", tenant_id="t1", email="viewer@test.com",
            role=UserRole.VIEWER,
        )

        dep = require_superadmin()
        with pytest.raises(AuthError, match="Superadmin access required"):
            await dep(user=user)

    @pytest.mark.asyncio
    async def test_rejects_agent_user(self) -> None:
        """Agent user should be rejected by require_superadmin."""
        from app.core.dependencies import require_superadmin

        user = User(
            id="ag-1", tenant_id="t1", email="agent@test.com",
            role=UserRole.AGENT,
        )

        dep = require_superadmin()
        with pytest.raises(AuthError, match="Superadmin access required"):
            await dep(user=user)

    async def _get_user(self, credentials, session, settings):  # noqa: ANN001
        """Helper to call get_current_user with mocked deps."""
        from app.core.dependencies import get_current_user

        return await get_current_user(
            credentials=credentials, session=session, settings=settings,
        )


# ═══════════════════════════════════════════════════════════════════
# Task 3: require_role() accepts superadmin when ADMIN is allowed
# ═══════════════════════════════════════════════════════════════════


class TestRequireRoleSuperadminCompat:
    """Verify require_role() treats superadmin as having admin privileges."""

    @pytest.mark.asyncio
    async def test_superadmin_passes_admin_role_check(self) -> None:
        """Superadmin should be accepted by require_role(UserRole.ADMIN)."""
        from app.core.dependencies import require_role

        user = User(
            id="sa-1", tenant_id="t1", email="sa@test.com",
            role=UserRole.SUPERADMIN,
        )
        dep = require_role(UserRole.ADMIN)
        result = await dep(user=user)
        assert result.role == UserRole.SUPERADMIN

    @pytest.mark.asyncio
    async def test_superadmin_fails_viewer_only_check(self) -> None:
        """Superadmin should NOT pass require_role(UserRole.VIEWER) alone."""
        from app.core.dependencies import require_role

        user = User(
            id="sa-1", tenant_id="t1", email="sa@test.com",
            role=UserRole.SUPERADMIN,
        )
        dep = require_role(UserRole.VIEWER)
        with pytest.raises(AuthError, match="Insufficient permissions"):
            await dep(user=user)

    @pytest.mark.asyncio
    async def test_superadmin_passes_agent_admin_check(self) -> None:
        """Superadmin should pass require_role(UserRole.ADMIN, UserRole.AGENT)."""
        from app.core.dependencies import require_role

        user = User(
            id="sa-1", tenant_id="t1", email="sa@test.com",
            role=UserRole.SUPERADMIN,
        )
        dep = require_role(UserRole.ADMIN, UserRole.AGENT)
        result = await dep(user=user)
        assert result.role == UserRole.SUPERADMIN

    @pytest.mark.asyncio
    async def test_regular_admin_still_works(self) -> None:
        """Regular admin should still pass require_role(UserRole.ADMIN)."""
        from app.core.dependencies import require_role

        user = User(
            id="a-1", tenant_id="t1", email="admin@test.com",
            role=UserRole.ADMIN,
        )
        dep = require_role(UserRole.ADMIN)
        result = await dep(user=user)
        assert result.role == UserRole.ADMIN


# ═══════════════════════════════════════════════════════════════════
# Task 5: Superadmin self-registration blocked
# ═══════════════════════════════════════════════════════════════════


class TestSuperadminRegistrationBlocked:
    """Verify superadmin role cannot be self-registered."""

    @pytest.mark.asyncio
    async def test_register_with_superadmin_role_rejected(self, client) -> None:  # noqa: ANN001
        """POST /api/v1/auth/register with role=superadmin should return 422."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "attacker@test.com",
                "password": "StrongP@ss1",
                "tenant_id": "some-tenant",
                "role": "superadmin",
            },
        )
        assert response.status_code == 422
        data = response.json()
        assert "superadmin" in data["error"]["message"].lower()
