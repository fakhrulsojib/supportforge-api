"""Repository interface definitions (ports).

Abstract base classes that define how domain services interact with
data persistence. Concrete implementations live in infrastructure/.

NO framework imports allowed — these are pure Python ABCs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from app.domain.models.conversation import Conversation, Message
    from app.domain.models.document import Document, DocumentChunk
    from app.domain.models.enums import (
        ConversationStatus,
        DocumentStatus,
        EscalationTrigger,
        FailureReason,
        FeedbackType,
        TenantStatus,
    )
    from app.domain.models.failed_query import FailedQuery
    from app.domain.models.tenant import Tenant, TenantCreate
    from app.domain.models.user import User, UserCreate


class TenantRepository(ABC):
    """Port for tenant data persistence."""

    @abstractmethod
    async def create(self, tenant: TenantCreate) -> Tenant: ...

    @abstractmethod
    async def get_by_id(self, tenant_id: str) -> Tenant | None: ...

    @abstractmethod
    async def get_by_slug(self, slug: str) -> Tenant | None: ...

    @abstractmethod
    async def list_all(self) -> list[Tenant]: ...

    @abstractmethod
    async def list_all_with_status(
        self,
        *,
        status: TenantStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Tenant]: ...

    @abstractmethod
    async def count_all(self, *, status: TenantStatus | None = None) -> int: ...

    @abstractmethod
    async def update(self, tenant_id: str, **kwargs: object) -> Tenant | None: ...

    @abstractmethod
    async def update_status(self, tenant_id: str, status: TenantStatus) -> Tenant | None: ...

    @abstractmethod
    async def delete(self, tenant_id: str) -> bool: ...


class UserRepository(ABC):
    """Port for user data persistence."""

    @abstractmethod
    async def create(self, tenant_id: str, user: UserCreate, password_hash: str) -> User: ...

    @abstractmethod
    async def get_by_id(self, user_id: str) -> User | None: ...

    @abstractmethod
    async def get_by_email(self, email: str, tenant_id: str) -> User | None: ...

    @abstractmethod
    async def list_by_tenant(self, tenant_id: str) -> list[User]: ...


class ConversationRepository(ABC):
    """Port for conversation data persistence."""

    @abstractmethod
    async def create(self, tenant_id: str, user_id: str, conversation_id: str = "") -> Conversation: ...

    @abstractmethod
    async def get_by_id(self, conversation_id: str) -> Conversation | None: ...

    @abstractmethod
    async def list_by_tenant(self, tenant_id: str, limit: int = 50, offset: int = 0) -> list[Conversation]: ...

    @abstractmethod
    async def list_by_user(
        self, tenant_id: str, user_id: str, limit: int = 50, offset: int = 0,
    ) -> list[Conversation]: ...

    @abstractmethod
    async def update_status(self, conversation_id: str, status: ConversationStatus) -> Conversation | None: ...

    @abstractmethod
    async def update_escalation_trigger(
        self, conversation_id: str, trigger: EscalationTrigger,
    ) -> Conversation | None: ...

    @abstractmethod
    async def list_escalated(
        self,
        tenant_id: str,
        *,
        trigger: EscalationTrigger | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Conversation], int]: ...

    @abstractmethod
    async def count_open_escalations(self, tenant_id: str) -> int: ...


class MessageRepository(ABC):
    """Port for message data persistence."""

    @abstractmethod
    async def create(self, message: Message) -> Message: ...

    @abstractmethod
    async def get_by_id(self, message_id: str) -> Message | None: ...

    @abstractmethod
    async def list_by_conversation(self, conversation_id: str, limit: int = 100) -> list[Message]: ...

    @abstractmethod
    async def update_feedback(self, message_id: str, feedback: FeedbackType) -> Message | None: ...

    @abstractmethod
    async def update_review_status(self, message_id: str, reviewed_by: str) -> Message | None: ...

    @abstractmethod
    async def get_preceding_user_message(
        self, conversation_id: str, assistant_message_id: str,
    ) -> Message | None: ...

    @abstractmethod
    async def list_negative_feedback(
        self,
        tenant_id: str,
        *,
        reviewed: bool | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Message], int]: ...

    @abstractmethod
    async def list_flagged_messages(
        self,
        tenant_id: str,
        *,
        reviewed: bool | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Message], int]: ...

    @abstractmethod
    async def count_unreviewed_negative(self, tenant_id: str) -> int: ...

    @abstractmethod
    async def count_unreviewed_flagged(self, tenant_id: str) -> int: ...


class DocumentRepository(ABC):
    """Port for document data persistence."""

    @abstractmethod
    async def create(self, document: Document) -> Document: ...

    @abstractmethod
    async def get_by_id(self, document_id: str) -> Document | None: ...

    @abstractmethod
    async def list_by_tenant(self, tenant_id: str) -> list[Document]: ...

    @abstractmethod
    async def update_status(
        self, document_id: str, status: DocumentStatus, chunk_count: int = 0
    ) -> Document | None: ...

    @abstractmethod
    async def delete(self, document_id: str) -> bool: ...

    @abstractmethod
    async def create_chunk(self, chunk: DocumentChunk) -> DocumentChunk: ...

    @abstractmethod
    async def get_chunks_by_document(self, document_id: str) -> list[DocumentChunk]: ...

    @abstractmethod
    async def delete_chunks_by_document(self, document_id: str) -> int: ...


class FailedQueryRepository(ABC):
    """Port for failed query data persistence."""

    @abstractmethod
    async def create(self, failed_query: FailedQuery) -> FailedQuery: ...

    @abstractmethod
    async def get_by_id(self, query_id: str) -> FailedQuery | None: ...

    @abstractmethod
    async def list_by_tenant(
        self,
        tenant_id: str,
        *,
        failure_reason: FailureReason | None = None,
        resolved: bool | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[FailedQuery], int]: ...

    @abstractmethod
    async def mark_resolved(
        self, query_id: str, resolved_by: str,
    ) -> FailedQuery | None: ...

    @abstractmethod
    async def count_unresolved(self, tenant_id: str) -> int: ...

    @abstractmethod
    async def get_stats(self, tenant_id: str) -> dict[str, Any]: ...
