"""Tests for custom exception hierarchy."""

from __future__ import annotations

from app.core.exceptions import (
    AuthError,
    ConversationNotFoundError,
    DocumentNotFoundError,
    IngestionError,
    LLMError,
    RateLimitError,
    SupportForgeError,
    TenantNotFoundError,
)


class TestSupportForgeError:
    """Test suite for base exception."""

    def test_default_values(self) -> None:
        """Base error should have sensible defaults."""
        err = SupportForgeError()
        assert err.message == "An unexpected error occurred"
        assert err.status_code == 500
        assert err.error_code == "INTERNAL_ERROR"
        assert str(err) == "An unexpected error occurred"

    def test_custom_values(self) -> None:
        """Base error should accept custom values."""
        err = SupportForgeError(message="Custom", status_code=418, error_code="TEAPOT")
        assert err.message == "Custom"
        assert err.status_code == 418
        assert err.error_code == "TEAPOT"

    def test_is_exception(self) -> None:
        """SupportForgeError should be a proper Exception subclass."""
        err = SupportForgeError()
        assert isinstance(err, Exception)


class TestTenantNotFoundError:
    """Test suite for TenantNotFoundError."""

    def test_with_tenant_id(self) -> None:
        err = TenantNotFoundError("tenant-123")
        assert err.status_code == 404
        assert err.error_code == "TENANT_NOT_FOUND"
        assert "tenant-123" in err.message

    def test_without_tenant_id(self) -> None:
        err = TenantNotFoundError()
        assert err.status_code == 404
        assert "Tenant not found" in err.message

    def test_is_supportforge_error(self) -> None:
        assert isinstance(TenantNotFoundError(), SupportForgeError)


class TestDocumentNotFoundError:
    """Test suite for DocumentNotFoundError."""

    def test_with_document_id(self) -> None:
        err = DocumentNotFoundError("doc-456")
        assert err.status_code == 404
        assert "doc-456" in err.message

    def test_without_document_id(self) -> None:
        err = DocumentNotFoundError()
        assert "Document not found" in err.message


class TestConversationNotFoundError:
    """Test suite for ConversationNotFoundError."""

    def test_with_conversation_id(self) -> None:
        err = ConversationNotFoundError("conv-789")
        assert err.status_code == 404
        assert "conv-789" in err.message


class TestIngestionError:
    """Test suite for IngestionError."""

    def test_defaults(self) -> None:
        err = IngestionError()
        assert err.status_code == 422
        assert err.error_code == "INGESTION_ERROR"

    def test_custom_message(self) -> None:
        err = IngestionError("PDF parsing failed")
        assert err.message == "PDF parsing failed"


class TestLLMError:
    """Test suite for LLMError."""

    def test_defaults(self) -> None:
        err = LLMError()
        assert err.status_code == 502
        assert err.error_code == "LLM_ERROR"


class TestAuthError:
    """Test suite for AuthError."""

    def test_defaults(self) -> None:
        err = AuthError()
        assert err.status_code == 401
        assert err.error_code == "AUTH_ERROR"


class TestRateLimitError:
    """Test suite for RateLimitError."""

    def test_defaults(self) -> None:
        err = RateLimitError()
        assert err.status_code == 429
        assert err.error_code == "RATE_LIMIT_EXCEEDED"
        assert err.retry_after == 60

    def test_custom_retry_after(self) -> None:
        err = RateLimitError(retry_after=120)
        assert err.retry_after == 120
