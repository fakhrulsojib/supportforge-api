"""Domain enumerations used across the application.

These are pure Python enums — NO framework imports allowed in the domain layer.
"""

from __future__ import annotations

import enum


class UserRole(str, enum.Enum):
    """Roles that a user can have within a tenant.

    SUPERADMIN is a platform-wide role — not scoped to any single tenant.
    It grants cross-tenant access and platform management capabilities.
    """

    ADMIN = "admin"
    AGENT = "agent"
    VIEWER = "viewer"
    SUPERADMIN = "superadmin"


class ConversationStatus(str, enum.Enum):
    """Lifecycle status of a conversation."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class MessageRole(str, enum.Enum):
    """Who sent a message in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class FeedbackType(str, enum.Enum):
    """User feedback on an assistant message."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NONE = "none"


class DocumentStatus(str, enum.Enum):
    """Processing status of an uploaded document."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class ValidationStatus(str, enum.Enum):
    """Output validation status of an assistant message.

    Tracks whether a message passed post-generation validation
    checks (anti-hallucination guard).
    """

    PASSED = "passed"
    FLAGGED = "flagged"
    NONE = "none"


class EscalationTrigger(str, enum.Enum):
    """What triggered an escalation to a human agent.

    Tracks the specific detection method that caused the conversation
    to be escalated, for analytics and review purposes.
    """

    NONE = "none"
    NO_CONTEXT = "no_context"
    SENTIMENT = "sentiment"
    REPETITION = "repetition"
    EXPLICIT_REQUEST = "explicit_request"


class TenantStatus(str, enum.Enum):
    """Lifecycle status of a tenant.

    Controls whether a tenant's users can access services like chat.
    Suspended and archived tenants are blocked from chat (both REST and WS).
    """

    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"
