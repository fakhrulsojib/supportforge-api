"""Tests for domain models and enums."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.models.conversation import Conversation, Message
from app.domain.models.document import Document, DocumentChunk
from app.domain.models.enums import (
    ConversationStatus,
    DocumentStatus,
    FeedbackType,
    MessageRole,
    UserRole,
    ValidationStatus,
)
from app.domain.models.tenant import Tenant, TenantCreate
from app.domain.models.user import User, UserCreate


class TestEnums:
    """Test suite for domain enums."""

    def test_user_role_values(self) -> None:
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.AGENT.value == "agent"
        assert UserRole.VIEWER.value == "viewer"

    def test_conversation_status_values(self) -> None:
        assert ConversationStatus.ACTIVE.value == "active"
        assert ConversationStatus.RESOLVED.value == "resolved"
        assert ConversationStatus.ESCALATED.value == "escalated"

    def test_message_role_values(self) -> None:
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.SYSTEM.value == "system"

    def test_feedback_type_values(self) -> None:
        assert FeedbackType.POSITIVE.value == "positive"
        assert FeedbackType.NEGATIVE.value == "negative"
        assert FeedbackType.NONE.value == "none"

    def test_document_status_values(self) -> None:
        assert DocumentStatus.PENDING.value == "pending"
        assert DocumentStatus.PROCESSING.value == "processing"
        assert DocumentStatus.READY.value == "ready"
        assert DocumentStatus.FAILED.value == "failed"

    def test_validation_status_values(self) -> None:
        assert ValidationStatus.PASSED.value == "passed"
        assert ValidationStatus.FLAGGED.value == "flagged"
        assert ValidationStatus.NONE.value == "none"

    def test_enums_are_string_enums(self) -> None:
        """All enums should be str enums for JSON serialization."""
        assert isinstance(UserRole.ADMIN, str)
        assert isinstance(ConversationStatus.ACTIVE, str)
        assert isinstance(MessageRole.USER, str)
        assert isinstance(FeedbackType.POSITIVE, str)
        assert isinstance(DocumentStatus.PENDING, str)
        assert isinstance(ValidationStatus.NONE, str)


class TestTenantModel:
    """Test suite for Tenant domain model."""

    def test_valid_tenant(self) -> None:
        tenant = Tenant(name="Acme Store", slug="acme-store")
        assert tenant.name == "Acme Store"
        assert tenant.slug == "acme-store"
        assert tenant.config_json == {}

    def test_tenant_with_config(self) -> None:
        config = {"chat_model": "llama3", "temperature": 0.7}
        tenant = Tenant(name="Test", slug="test-co", config_json=config)
        assert tenant.config_json["chat_model"] == "llama3"

    def test_tenant_slug_pattern_valid(self) -> None:
        """Slug must match pattern: lowercase alphanumeric with hyphens."""
        for slug in ("ab", "acme-store", "test-123-co"):
            TenantCreate(name="Test", slug=slug)

    def test_tenant_slug_pattern_invalid(self) -> None:
        """Invalid slugs should be rejected."""
        for slug in ("a", "-start", "end-", "UPPER", "has space"):
            with pytest.raises(ValidationError):
                TenantCreate(name="Test", slug=slug)

    def test_tenant_name_required(self) -> None:
        with pytest.raises(ValidationError):
            TenantCreate(name="", slug="test-co")


class TestUserModel:
    """Test suite for User domain model."""

    def test_valid_user(self) -> None:
        user = User(email="test@example.com", role=UserRole.ADMIN)
        assert user.email == "test@example.com"
        assert user.role == UserRole.ADMIN

    def test_user_default_role(self) -> None:
        user = User(email="test@example.com")
        assert user.role == UserRole.VIEWER

    def test_user_create_email_required(self) -> None:
        """UserCreate should reject empty email."""
        with pytest.raises(ValidationError):
            UserCreate(email="")

    def test_user_create_valid(self) -> None:
        uc = UserCreate(email="test@example.com")
        assert uc.email == "test@example.com"
        assert uc.role == UserRole.VIEWER


class TestConversationModel:
    """Test suite for Conversation domain model."""

    def test_default_status(self) -> None:
        conv = Conversation()
        assert conv.status == ConversationStatus.ACTIVE

    def test_conversation_with_status(self) -> None:
        conv = Conversation(status=ConversationStatus.ESCALATED)
        assert conv.status == ConversationStatus.ESCALATED


class TestMessageModel:
    """Test suite for Message domain model."""

    def test_valid_message(self) -> None:
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.feedback == FeedbackType.NONE
        assert msg.validation_status == ValidationStatus.NONE

    def test_message_content_required(self) -> None:
        with pytest.raises(ValidationError):
            Message(role=MessageRole.USER, content="")

    def test_message_with_sources(self) -> None:
        sources = [{"doc": "faq.pdf", "chunk": 3}]
        msg = Message(role=MessageRole.ASSISTANT, content="Answer", sources_json=sources)
        assert len(msg.sources_json) == 1

    def test_moderation_fields_default_empty(self) -> None:
        """Moderation fields should default to empty strings."""
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.moderation_reason == ""
        assert msg.moderation_matched_term == ""

    def test_moderation_fields_set(self) -> None:
        """Moderation fields should accept values."""
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Blocked response",
            validation_status=ValidationStatus.FLAGGED,
            moderation_reason="jailbreak_detected",
            moderation_matched_term="ignore previous instructions",
        )
        assert msg.moderation_reason == "jailbreak_detected"
        assert msg.moderation_matched_term == "ignore previous instructions"
        assert msg.validation_status == ValidationStatus.FLAGGED



class TestDocumentModel:
    """Test suite for Document domain model."""

    def test_valid_document(self) -> None:
        doc = Document(filename="faq.pdf", file_type="pdf")
        assert doc.filename == "faq.pdf"
        assert doc.status == DocumentStatus.PENDING

    def test_document_filename_required(self) -> None:
        with pytest.raises(ValidationError):
            Document(filename="", file_type="pdf")


class TestDocumentChunkModel:
    """Test suite for DocumentChunk domain model."""

    def test_valid_chunk(self) -> None:
        chunk = DocumentChunk(
            document_id="doc-1",
            chunk_index=0,
            content="Some text content",
            chroma_id="chroma-123",
        )
        assert chunk.chunk_index == 0
        assert chunk.content == "Some text content"
