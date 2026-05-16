"""Cryptographic helpers for field-level encryption of tenant secrets.

Provides AES-256 encryption via Fernet for storing sensitive values
(e.g. API keys) in ``config_json``.  The application ``secret_key``
from Settings is used to derive a Fernet-compatible key via SHA-256.

Usage::

    from app.core.crypto import encrypt_value, decrypt_value, mask_api_key

    encrypted = encrypt_value("AIzaSy...", settings.secret_key)
    decrypted = decrypt_value(encrypted, settings.secret_key)
    masked    = mask_api_key("AIzaSy...")  # → "AIza...****"

Security notes:
    - Keys are encrypted at write time and decrypted at read time.
    - Encrypted values are never logged or exposed in API responses.
    - The ``mask_api_key`` helper produces a safe display preview.
"""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken


def derive_fernet_key(secret: str) -> bytes:
    """Derive a 32-byte Fernet key from an arbitrary-length secret string.

    Uses SHA-256 to produce a deterministic 32-byte digest, then
    base64url-encodes it to satisfy Fernet's key format requirement.

    Args:
        secret: The application secret key (any length).

    Returns:
        A 44-byte url-safe base64-encoded key suitable for ``Fernet()``.
    """
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def encrypt_value(plaintext: str, key: str) -> str:
    """Encrypt a plaintext string using Fernet (AES-128-CBC + HMAC).

    Args:
        plaintext: The value to encrypt (e.g. an API key).
        key: The application secret key used for key derivation.

    Returns:
        Base64-encoded ciphertext string safe for JSON storage.
    """
    fernet_key = derive_fernet_key(key)
    f = Fernet(fernet_key)
    return f.encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_value(ciphertext: str, key: str) -> str:
    """Decrypt a Fernet-encrypted ciphertext string.

    Args:
        ciphertext: The encrypted value from ``encrypt_value()``.
        key: The same application secret key used for encryption.

    Returns:
        The original plaintext string.

    Raises:
        ValueError: If decryption fails (wrong key, corrupted data,
            or tampered ciphertext).
    """
    if not ciphertext:
        msg = "Cannot decrypt empty ciphertext"
        raise ValueError(msg)

    fernet_key = derive_fernet_key(key)
    f = Fernet(fernet_key)
    try:
        return f.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except (InvalidToken, Exception) as exc:
        msg = f"Failed to decrypt value: {type(exc).__name__}"
        raise ValueError(msg) from exc


def mask_api_key(api_key: str) -> str:
    """Produce a masked display preview of an API key.

    Shows the first 4 characters followed by ``...****``.
    Keys shorter than 4 characters show ``****`` only.
    Empty keys return an empty string.

    Args:
        api_key: The raw API key string.

    Returns:
        A masked preview string (e.g. ``"AIza...****"``).
    """
    if not api_key:
        return ""
    if len(api_key) <= 4:
        return "****"
    return f"{api_key[:4]}...****"
