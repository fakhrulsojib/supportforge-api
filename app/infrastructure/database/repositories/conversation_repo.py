"""SQLAlchemy implementation of ConversationRepository and MessageRepository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from app.domain.interfaces.repository import ConversationRepository, MessageRepository
from app.domain.models.conversation import Conversation, Message
from app.domain.models.enums import ConversationStatus, EscalationTrigger, FeedbackType
from app.infrastructure.database.models import ConversationModel, MessageModel

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class SQLConversationRepository(ConversationRepository):
    """Concrete conversation repository backed by PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def _to_domain(self, model: ConversationModel) -> Conversation:
        """Convert ORM model to domain model."""
        return Conversation(
            id=model.id,
            tenant_id=model.tenant_id,
            user_id=model.user_id or "",
            started_at=model.started_at,
            ended_at=model.ended_at,
            status=model.status,
            escalation_trigger=model.escalation_trigger,
        )

    async def create(self, tenant_id: str, user_id: str, conversation_id: str = "") -> Conversation:
        """Create a new conversation.

        Args:
            tenant_id: Tenant owning the conversation.
            user_id: User who started the conversation.
            conversation_id: Optional pre-assigned UUID. If empty, the
                ORM default (``uuid4``) generates one automatically.
        """
        model = ConversationModel(
            tenant_id=tenant_id,
            user_id=user_id or None,
        )
        if conversation_id:
            model.id = conversation_id
        self._session.add(model)
        await self._session.flush()
        return self._to_domain(model)

    async def get_by_id(self, conversation_id: str) -> Conversation | None:
        """Get a conversation by ID."""
        result = await self._session.get(ConversationModel, conversation_id)
        return self._to_domain(result) if result else None

    async def list_by_tenant(self, tenant_id: str, limit: int = 50, offset: int = 0) -> list[Conversation]:
        """List conversations for a tenant with pagination."""
        stmt = (
            select(ConversationModel)
            .where(ConversationModel.tenant_id == tenant_id)
            .order_by(ConversationModel.started_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return [self._to_domain(m) for m in result.scalars().all()]

    async def list_by_user(self, tenant_id: str, user_id: str, limit: int = 50, offset: int = 0) -> list[Conversation]:
        """List conversations for a specific user within a tenant."""
        stmt = (
            select(ConversationModel)
            .where(
                ConversationModel.tenant_id == tenant_id,
                ConversationModel.user_id == user_id,
            )
            .order_by(ConversationModel.started_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return [self._to_domain(m) for m in result.scalars().all()]

    async def update_status(self, conversation_id: str, status: ConversationStatus) -> Conversation | None:
        """Update a conversation's status."""
        model = await self._session.get(ConversationModel, conversation_id)
        if not model:
            return None
        model.status = status
        await self._session.flush()
        return self._to_domain(model)

    async def update_escalation_trigger(
        self, conversation_id: str, trigger: EscalationTrigger,
    ) -> Conversation | None:
        """Update a conversation's escalation trigger."""
        model = await self._session.get(ConversationModel, conversation_id)
        if not model:
            return None
        model.escalation_trigger = trigger
        await self._session.flush()
        return self._to_domain(model)


class SQLMessageRepository(MessageRepository):
    """Concrete message repository backed by PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def _to_domain(self, model: MessageModel) -> Message:
        """Convert ORM model to domain model."""
        return Message(
            id=model.id,
            conversation_id=model.conversation_id,
            role=model.role,
            content=model.content,
            thinking=model.thinking,
            sources_json=model.sources_json,
            model_used=model.model_used,
            tokens_in=model.tokens_in,
            tokens_out=model.tokens_out,
            feedback=model.feedback,
            validation_status=model.validation_status,
            moderation_reason=model.moderation_reason,
            moderation_matched_term=model.moderation_matched_term,
            created_at=model.created_at,
        )

    async def create(self, message: Message) -> Message:
        """Create a new message."""
        model = MessageModel(
            conversation_id=message.conversation_id,
            role=message.role,
            content=message.content,
            thinking=message.thinking,
            sources_json=message.sources_json,
            model_used=message.model_used,
            tokens_in=message.tokens_in,
            tokens_out=message.tokens_out,
            feedback=message.feedback,
            validation_status=message.validation_status,
            moderation_reason=message.moderation_reason,
            moderation_matched_term=message.moderation_matched_term,
        )
        self._session.add(model)
        await self._session.flush()
        return self._to_domain(model)

    async def get_by_id(self, message_id: str) -> Message | None:
        """Get a message by ID."""
        result = await self._session.get(MessageModel, message_id)
        return self._to_domain(result) if result else None

    async def list_by_conversation(self, conversation_id: str, limit: int = 100) -> list[Message]:
        """List messages in a conversation ordered by creation time."""
        stmt = (
            select(MessageModel)
            .where(MessageModel.conversation_id == conversation_id)
            .order_by(MessageModel.created_at.asc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [self._to_domain(m) for m in result.scalars().all()]

    async def update_feedback(self, message_id: str, feedback: FeedbackType) -> Message | None:
        """Update feedback on a message."""
        model = await self._session.get(MessageModel, message_id)
        if not model:
            return None
        model.feedback = feedback
        await self._session.flush()
        return self._to_domain(model)
