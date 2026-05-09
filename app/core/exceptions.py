"""Custom exception hierarchy for SupportForge.

Every custom exception maps to a specific HTTP status code and is
caught by FastAPI exception handlers in main.py to return a consistent
JSON error response.
"""

from __future__ import annotations


class SupportForgeError(Exception):
    """Base exception for all SupportForge errors.

    Attributes:
        message: Human-readable error description.
        status_code: HTTP status code to return (default 500).
        error_code: Machine-readable error identifier.
    """

    def __init__(
        self,
        message: str = "An unexpected error occurred",
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.message)


class TenantNotFoundError(SupportForgeError):
    """Raised when a tenant cannot be found."""

    def __init__(self, tenant_id: str = "") -> None:
        detail = f"Tenant '{tenant_id}' not found" if tenant_id else "Tenant not found"
        super().__init__(
            message=detail,
            status_code=404,
            error_code="TENANT_NOT_FOUND",
        )


class DocumentNotFoundError(SupportForgeError):
    """Raised when a document cannot be found."""

    def __init__(self, document_id: str = "") -> None:
        detail = f"Document '{document_id}' not found" if document_id else "Document not found"
        super().__init__(
            message=detail,
            status_code=404,
            error_code="DOCUMENT_NOT_FOUND",
        )


class ConversationNotFoundError(SupportForgeError):
    """Raised when a conversation cannot be found."""

    def __init__(self, conversation_id: str = "") -> None:
        detail = f"Conversation '{conversation_id}' not found" if conversation_id else "Conversation not found"
        super().__init__(
            message=detail,
            status_code=404,
            error_code="CONVERSATION_NOT_FOUND",
        )


class IngestionError(SupportForgeError):
    """Raised when document ingestion fails."""

    def __init__(self, message: str = "Document ingestion failed") -> None:
        super().__init__(
            message=message,
            status_code=422,
            error_code="INGESTION_ERROR",
        )


class LLMError(SupportForgeError):
    """Raised when LLM communication fails."""

    def __init__(self, message: str = "LLM service unavailable") -> None:
        super().__init__(
            message=message,
            status_code=502,
            error_code="LLM_ERROR",
        )


class AuthError(SupportForgeError):
    """Raised for authentication/authorization failures."""

    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTH_ERROR",
        )


class RateLimitError(SupportForgeError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60) -> None:
        self.retry_after = retry_after
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
        )


class TenantSuspendedError(SupportForgeError):
    """Raised when a non-active tenant tries to access services."""

    def __init__(self, tenant_id: str = "") -> None:
        self.tenant_id = tenant_id
        detail = (
            "Your organization's account is currently suspended. "
            "Please contact your administrator."
        )
        super().__init__(
            message=detail,
            status_code=403,
            error_code="TENANT_SUSPENDED",
        )
