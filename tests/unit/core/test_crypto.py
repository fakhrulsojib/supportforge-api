"""Unit tests for the crypto utility module.

Covers:
    - Encrypt/decrypt round-trip with valid key
    - Different plaintexts produce different ciphertexts
    - Decryption with wrong key fails
    - Empty plaintext encrypts/decrypts correctly
    - Unicode plaintext encrypts/decrypts correctly
    - Long API key strings survive round-trip
    - mask_api_key masks correctly
    - mask_api_key with short keys
    - mask_api_key with empty string
    - derive_fernet_key produces valid Fernet key from arbitrary secret
    - derive_fernet_key is deterministic
    - decrypt with corrupted ciphertext raises ValueError
"""

from __future__ import annotations

import pytest

from app.core.crypto import decrypt_value, derive_fernet_key, encrypt_value, mask_api_key


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def secret_key() -> str:
    """A sample application secret key."""
    return "change-me-to-a-random-secret-key"


@pytest.fixture
def alt_secret_key() -> str:
    """A different secret key for negative tests."""
    return "another-completely-different-key"


@pytest.fixture
def sample_api_key() -> str:
    """A realistic Gemini API key."""
    return "AIzaSyD-fake-test-key-1234567890abcdef"


# ── Round-Trip Encryption ───────────────────────────────────────


class TestEncryptDecryptRoundTrip:
    """Tests for encrypt → decrypt round-trip consistency."""

    def test_round_trip_basic(self, secret_key: str, sample_api_key: str) -> None:
        """Encrypt then decrypt should return the original value."""
        encrypted = encrypt_value(sample_api_key, secret_key)
        decrypted = decrypt_value(encrypted, secret_key)
        assert decrypted == sample_api_key

    def test_encrypted_differs_from_plaintext(
        self, secret_key: str, sample_api_key: str
    ) -> None:
        """Encrypted output must not be the same as the input."""
        encrypted = encrypt_value(sample_api_key, secret_key)
        assert encrypted != sample_api_key

    def test_different_plaintexts_different_ciphertexts(
        self, secret_key: str
    ) -> None:
        """Two different plaintexts should produce different ciphertexts."""
        enc_a = encrypt_value("key-alpha", secret_key)
        enc_b = encrypt_value("key-bravo", secret_key)
        assert enc_a != enc_b

    def test_empty_plaintext(self, secret_key: str) -> None:
        """Empty string should encrypt and decrypt correctly."""
        encrypted = encrypt_value("", secret_key)
        decrypted = decrypt_value(encrypted, secret_key)
        assert decrypted == ""

    def test_unicode_plaintext(self, secret_key: str) -> None:
        """Unicode characters should survive encryption round-trip."""
        value = "APIকী-🔑-テスト"
        encrypted = encrypt_value(value, secret_key)
        decrypted = decrypt_value(encrypted, secret_key)
        assert decrypted == value

    def test_long_api_key(self, secret_key: str) -> None:
        """Long API key strings should survive round-trip."""
        long_key = "AIza" + "x" * 500
        encrypted = encrypt_value(long_key, secret_key)
        decrypted = decrypt_value(encrypted, secret_key)
        assert decrypted == long_key


# ── Wrong Key Decryption ────────────────────────────────────────


class TestWrongKeyDecryption:
    """Tests for decryption failure with incorrect key."""

    def test_wrong_key_raises(
        self,
        secret_key: str,
        alt_secret_key: str,
        sample_api_key: str,
    ) -> None:
        """Decrypting with a different key must raise ValueError."""
        encrypted = encrypt_value(sample_api_key, secret_key)
        with pytest.raises(ValueError, match="[Dd]ecrypt"):
            decrypt_value(encrypted, alt_secret_key)

    def test_corrupted_ciphertext_raises(self, secret_key: str) -> None:
        """Corrupted ciphertext should raise ValueError."""
        with pytest.raises(ValueError, match="[Dd]ecrypt"):
            decrypt_value("not-valid-base64-ciphertext!!!", secret_key)

    def test_empty_ciphertext_raises(self, secret_key: str) -> None:
        """Empty ciphertext should raise ValueError."""
        with pytest.raises(ValueError, match="[Dd]ecrypt"):
            decrypt_value("", secret_key)


# ── API Key Masking ─────────────────────────────────────────────


class TestMaskApiKey:
    """Tests for mask_api_key display masking."""

    def test_mask_standard_key(self, sample_api_key: str) -> None:
        """Standard API key should show prefix + masked suffix."""
        masked = mask_api_key(sample_api_key)
        assert masked.startswith(sample_api_key[:4])
        assert masked.endswith("****")
        assert len(masked) < len(sample_api_key)

    def test_mask_short_key(self) -> None:
        """Very short key should still produce a masked result."""
        masked = mask_api_key("abc")
        assert "****" in masked

    def test_mask_empty_key(self) -> None:
        """Empty key should return empty string."""
        assert mask_api_key("") == ""


# ── Key Derivation ──────────────────────────────────────────────


class TestDeriveKey:
    """Tests for derive_fernet_key determinism and validity."""

    def test_deterministic(self, secret_key: str) -> None:
        """Same input produces same derived key."""
        k1 = derive_fernet_key(secret_key)
        k2 = derive_fernet_key(secret_key)
        assert k1 == k2

    def test_different_inputs_different_keys(
        self, secret_key: str, alt_secret_key: str
    ) -> None:
        """Different secrets produce different derived keys."""
        k1 = derive_fernet_key(secret_key)
        k2 = derive_fernet_key(alt_secret_key)
        assert k1 != k2

    def test_valid_fernet_key_format(self, secret_key: str) -> None:
        """Derived key must be usable by Fernet (44 url-safe base64 chars)."""
        from cryptography.fernet import Fernet

        key = derive_fernet_key(secret_key)
        # Should not raise
        Fernet(key)
