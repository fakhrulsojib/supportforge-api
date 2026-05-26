"""Repository for tenant secrets — encrypted credential storage.

Secrets are stored as Fernet-encrypted strings in a separate table
(``tenant_secrets``), isolated from ``config_json``.  The GET API
returns only key names, never decrypted values.
"""

from __future__ import annotations

from typing import Any

import structlog
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.crypto import decrypt_value, encrypt_value
from app.infrastructure.database.models import TenantSecretModel

logger = structlog.get_logger(__name__)


class SQLTenantSecretRepository:
    """CRUD operations for tenant secrets."""

    def __init__(self, session: AsyncSession, *, encryption_key: str) -> None:
        self._session = session
        self._encryption_key = encryption_key

    async def upsert(self, tenant_id: str, *, key: str, value: str) -> None:
        """Create or update an encrypted secret.

        Args:
            tenant_id: Owning tenant.
            key: Secret name (e.g., ``tools_auth.default``).
            value: Plaintext value — will be encrypted before storage.
        """
        encrypted = encrypt_value(value, self._encryption_key)

        # Check if exists
        from sqlalchemy.exc import IntegrityError
        
        # Simple retry loop for TOCTOU race conditions during concurrent inserts
        for attempt in range(2):
            stmt = select(TenantSecretModel).where(
                TenantSecretModel.tenant_id == tenant_id,
                TenantSecretModel.key == key,
            )
            result = await self._session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                existing.encrypted_value = encrypted
                await self._session.flush()
                break
            else:
                try:
                    async with self._session.begin_nested():
                        model = TenantSecretModel(
                            tenant_id=tenant_id,
                            key=key,
                            encrypted_value=encrypted,
                        )
                        self._session.add(model)
                        await self._session.flush()
                    break
                except IntegrityError:
                    if attempt == 1:
                        raise
                    # SAVEPOINT rolled back — retry will find the existing record

    async def get_decrypted(self, tenant_id: str, key: str) -> str | None:
        """Get a single decrypted secret value.

        Returns None if the key doesn't exist or decryption fails
        (e.g., corrupted encrypted value or key mismatch).
        """
        stmt = select(TenantSecretModel).where(
            TenantSecretModel.tenant_id == tenant_id,
            TenantSecretModel.key == key,
        )
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        if not model:
            return None
        try:
            return decrypt_value(model.encrypted_value, self._encryption_key)
        except Exception:
            logger.warning(
                "secret_decrypt_failed",
                tenant_id=tenant_id,
                key=key,
            )
            return None

    async def get_all_decrypted(self, tenant_id: str) -> dict[str, str]:
        """Get all decrypted secrets for a tenant.

        Returns:
            Dict of ``{key: decrypted_value}``.
        """
        stmt = select(TenantSecretModel).where(
            TenantSecretModel.tenant_id == tenant_id
        )
        result = await self._session.execute(stmt)
        models = result.scalars().all()
        secrets: dict[str, str] = {}
        for model in models:
            try:
                secrets[model.key] = decrypt_value(
                    model.encrypted_value, self._encryption_key
                )
            except Exception:
                logger.warning(
                    "secret_decrypt_failed",
                    tenant_id=tenant_id,
                    key=model.key,
                )
        return secrets

    async def list_keys(self, tenant_id: str) -> list[str]:
        """List secret key names only — never returns values."""
        stmt = select(TenantSecretModel.key).where(
            TenantSecretModel.tenant_id == tenant_id
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def delete(self, tenant_id: str, key: str) -> bool:
        """Delete a secret. Returns True if deleted, False if not found."""
        stmt = (
            delete(TenantSecretModel)
            .where(
                TenantSecretModel.tenant_id == tenant_id,
                TenantSecretModel.key == key,
            )
        )
        result = await self._session.execute(stmt)
        await self._session.flush()
        return result.rowcount > 0
