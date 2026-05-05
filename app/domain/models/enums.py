"""Domain enumerations used across the application.

These are pure Python enums — NO framework imports allowed in the domain layer.
"""

from __future__ import annotations

import enum


class UserRole(str, enum.Enum):
    """Roles that a user can have within a tenant."""

    ADMIN = "admin"
    AGENT = "agent"
    VIEWER = "viewer"


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
