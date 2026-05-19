"""SQLAlchemy ORM models for all database tables.

These are infrastructure-layer models mapping to PostgreSQL tables.
They import from the domain enums but are NOT used in the domain layer.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.domain.models.enums import (
    ConversationStatus,
    DocumentStatus,
    EscalationTrigger,
    FailureReason,
    FeedbackType,
    MessageChannel,
    MessageRole,
    TenantStatus,
    UserRole,
    ValidationStatus,
)


def _enum_values(enum_cls: type) -> list[str]:
    """Extract enum `.value` strings for SQLAlchemy Enum column.

    By default SQLAlchemy uses enum member **names** (ACTIVE, PENDING) for
    PostgreSQL enum types.  Our Python enums use lowercase `.value` strings
    (active, pending) which must match ``server_default`` literals.
    """
    return [m.value for m in enum_cls]


def _generate_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class TenantModel(Base):
    """Tenant table — top-level entity for multi-tenancy."""

    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    config_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # type: ignore[assignment]
    status: Mapped[TenantStatus] = mapped_column(
        Enum(TenantStatus, values_callable=_enum_values), nullable=False,
        default=TenantStatus.ACTIVE, server_default="active",
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    # Relationships
    users: Mapped[list[UserModel]] = relationship("UserModel", back_populates="tenant", cascade="all, delete-orphan")
    conversations: Mapped[list[ConversationModel]] = relationship(
        "ConversationModel", back_populates="tenant", cascade="all, delete-orphan"
    )
    documents: Mapped[list[DocumentModel]] = relationship(
        "DocumentModel", back_populates="tenant", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_tenants_slug", "slug"),
        Index("ix_tenants_status", "status"),
    )


class UserModel(Base):
    """User table — belongs to a tenant."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    role: Mapped[UserRole] = mapped_column(Enum(UserRole, values_callable=_enum_values), nullable=False, default=UserRole.VIEWER)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    # Relationships
    tenant: Mapped[TenantModel] = relationship("TenantModel", back_populates="users")

    __table_args__ = (
        UniqueConstraint("email", "tenant_id", name="uq_users_email_tenant"),
        Index("ix_users_tenant_id", "tenant_id"),
        Index("ix_users_email", "email"),
    )


class ConversationModel(Base):
    """Conversation table — a chat session."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[ConversationStatus] = mapped_column(
        Enum(ConversationStatus, values_callable=_enum_values), nullable=False, default=ConversationStatus.ACTIVE
    )
    escalation_trigger: Mapped[EscalationTrigger] = mapped_column(
        Enum(EscalationTrigger, values_callable=_enum_values), nullable=False,
        default=EscalationTrigger.NONE, server_default="none",
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewed_by: Mapped[str] = mapped_column(String(36), nullable=False, default="")

    # Relationships
    tenant: Mapped[TenantModel] = relationship("TenantModel", back_populates="conversations")
    messages: Mapped[list[MessageModel]] = relationship(
        "MessageModel", back_populates="conversation", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_conversations_tenant_id", "tenant_id"),
        Index("ix_conversations_user_id", "user_id"),
    )


class MessageModel(Base):
    """Message table — a single message within a conversation."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole, values_callable=_enum_values), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    thinking: Mapped[str] = mapped_column(Text, nullable=False, default="")
    sources_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)  # type: ignore[assignment]
    model_used: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    tokens_in: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    feedback: Mapped[FeedbackType] = mapped_column(Enum(FeedbackType, values_callable=_enum_values), nullable=False, default=FeedbackType.NONE)
    validation_status: Mapped[ValidationStatus] = mapped_column(
        Enum(ValidationStatus, values_callable=_enum_values), nullable=False, default=ValidationStatus.NONE
    )
    moderation_reason: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    moderation_matched_term: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewed_by: Mapped[str] = mapped_column(String(36), nullable=False, default="")
    channel: Mapped[MessageChannel] = mapped_column(
        Enum(MessageChannel, values_callable=_enum_values),
        nullable=False,
        default=MessageChannel.TEXT,
        server_default="text",
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    # Relationships
    conversation: Mapped[ConversationModel] = relationship("ConversationModel", back_populates="messages")

    __table_args__ = (
        Index("ix_messages_conversation_id", "conversation_id"),
        Index("ix_messages_created_at", "created_at"),
        Index("ix_messages_validation_status", "validation_status"),
        Index("ix_messages_feedback", "feedback"),
        Index("ix_messages_channel", "channel"),
    )


class DocumentModel(Base):
    """Document table — uploaded files for RAG knowledge base."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[DocumentStatus] = mapped_column(Enum(DocumentStatus, values_callable=_enum_values), nullable=False, default=DocumentStatus.PENDING)
    uploaded_by: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    # Relationships
    tenant: Mapped[TenantModel] = relationship("TenantModel", back_populates="documents")
    chunks: Mapped[list[DocumentChunkModel]] = relationship(
        "DocumentChunkModel", back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_documents_tenant_id", "tenant_id"),)


class DocumentChunkModel(Base):
    """Document chunk table — individual chunks stored in vector DB."""

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chroma_id: Mapped[str] = mapped_column(String(100), nullable=False, default="")

    # Relationships
    document: Mapped[DocumentModel] = relationship("DocumentModel", back_populates="chunks")

    __table_args__ = (Index("ix_document_chunks_document_id", "document_id"),)


class DailyStatModel(Base):
    """Daily statistics table — aggregated analytics per tenant per day."""

    __tablename__ = "daily_stats"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    total_conversations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_messages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_satisfaction: Mapped[float | None] = mapped_column(nullable=True)
    top_intents_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # type: ignore[assignment]
    model_usage_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # type: ignore[assignment]

    __table_args__ = (
        Index("ix_daily_stats_tenant_id", "tenant_id"),
        UniqueConstraint("tenant_id", "date", name="uq_daily_stats_tenant_date"),
    )


class FailedQueryModel(Base):
    """Failed query table — tracks queries the RAG pipeline could not answer."""

    __tablename__ = "failed_queries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False,
    )
    message_id: Mapped[str] = mapped_column(String(36), nullable=False, default="")
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    failure_reason: Mapped[FailureReason] = mapped_column(
        Enum(FailureReason, values_callable=_enum_values), nullable=False,
    )
    retrieved_doc_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_relevance_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    escalation_trigger: Mapped[EscalationTrigger] = mapped_column(
        Enum(EscalationTrigger, values_callable=_enum_values), nullable=False,
        default=EscalationTrigger.NONE, server_default="none",
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_by: Mapped[str] = mapped_column(String(36), nullable=False, default="")

    # Relationships
    tenant: Mapped[TenantModel] = relationship("TenantModel")
    conversation: Mapped[ConversationModel] = relationship("ConversationModel")

    __table_args__ = (
        Index("ix_failed_queries_tenant_id", "tenant_id"),
        Index("ix_failed_queries_failure_reason", "failure_reason"),
        Index("ix_failed_queries_created_at", "created_at"),
    )
