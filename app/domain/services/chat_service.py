"""Chat service — orchestrates the RAG pipeline for conversation processing.

Relocated from ``app.api.v1.chat_service`` to the domain services layer
to maintain hexagonal architecture (domain logic should not live in API layer).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import structlog

from app.domain.models.enums import EscalationTrigger, FailureReason, ValidationStatus
from app.domain.services.content_moderator import ContentModerator
from app.domain.services.escalation_detector import EscalationDetector
from app.domain.services.output_validator import OutputValidator
from app.rag.graph import build_rag_graph
from app.rag.pipeline import (
    RAGState,
    escalation_node,
    generate_node,
    run_rag_pipeline,
)
from app.rag.prompt_builder import (
    build_rag_messages,
    build_system_prompt,
    format_rag_context,
)
from app.rag.tools.executor import ToolExecutor
from app.rag.tools.resolver import resolve_tenant_tools
from app.rag.tools.tool_loop import run_tool_loop

from app.core.event_hooks import EventType, HookPayload, dispatch_event

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from app.domain.interfaces.llm_provider import LLMProvider
    from app.domain.interfaces.vector_store import VectorStore
    from app.rag.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)

# ── Context-aware escalation messages ───────────────────────────

_ESCALATION_MESSAGES: dict[EscalationTrigger, str] = {
    EscalationTrigger.SENTIMENT: (
        "I can see this has been frustrating, and I sincerely apologize for the difficulty. "
        "Let me connect you with a specialist who can help resolve this directly. "
        "Please hold on — someone will be with you shortly."
    ),
    EscalationTrigger.REPETITION: (
        "I notice I haven't been able to fully address your question. "
        "Let me connect you with a team member who can give you a more thorough answer. "
        "Someone will be with you shortly."
    ),
    EscalationTrigger.EXPLICIT_REQUEST: (
        "Absolutely — I'll connect you with a human support agent right away. "
        "Someone will be with you shortly."
    ),
    EscalationTrigger.NO_CONTEXT: (
        "I wasn't able to find a confident answer to your question. "
        "I'm escalating this to a human support agent who will be able to help you. "
        "Please stand by — someone will be with you shortly."
    ),
    EscalationTrigger.LLM_DECISION: (
        "Based on what you've described, I believe a human agent would be best suited "
        "to help you with this. Let me connect you with someone from our team — "
        "they'll be with you shortly."
    ),
}

# Sentinel token the LLM can emit to trigger escalation
_ESCALATE_SENTINEL = "[ESCALATE]"

# Phrases in the LLM's *output* that indicate it could NOT answer the
# question from the provided context.
#
# Tier 1 — Admission of inability.  Always trigger, regardless of
#          response length.
# Tier 2 — Escalation-action phrases.  Only trigger when the response
#          is SHORT (< 200 chars), meaning the LLM gave no real answer.
#          Long responses containing these phrases are treated as polite
#          offers after a valid answer, NOT as real escalations.

_INABILITY_PATTERNS: tuple[str, ...] = (
    "i don't have that information",
    "i don't have enough information",
    "i wasn't able to find",
    "i couldn't find an answer",
    "i do not have that information",
    "i do not have enough information",
)

_ESCALATION_ACTION_PATTERNS: tuple[str, ...] = (
    "let me escalate",
    "i will escalate",
    "i'm escalating",
    "i am escalating",
    "i'll escalate",
    "escalate this to",
    "connect you with a",
)

# Short-response threshold — if the LLM produced fewer characters than
# this AND contains an escalation-action phrase, it is a real escalation.
_SHORT_RESPONSE_THRESHOLD = 200


def _detect_self_escalation(text: str) -> str | None:
    """Return the first matched self-escalation pattern, or None.

    Uses two-tier logic:
    - Tier 1 (inability phrases): always match.
    - Tier 2 (action phrases): only match when the response is short
      AND the sentence containing the phrase is NOT a question (ends
      with ``?``).  This prevents offers like *"Would you like me to
      escalate this to our team?"* from triggering.
    """
    lower = text.lower()

    # Tier 1 — always trigger
    for pattern in _INABILITY_PATTERNS:
        if pattern in lower:
            return pattern

    # Tier 2 — only trigger on short, non-question responses
    if len(text.strip()) < _SHORT_RESPONSE_THRESHOLD:
        import re
        # Split into sentences on . ! ?
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        for pattern in _ESCALATION_ACTION_PATTERNS:
            for sentence in sentences:
                if pattern in sentence.lower() and not sentence.rstrip().endswith("?"):
                    return pattern

    return None

# Static reply for messages received after a conversation is already escalated.
_POST_ESCALATION_REPLY = (
    "Your conversation has already been escalated to a support agent. "
    "They'll review everything you've shared here, including this message. "
    "Please hold tight — someone will be with you shortly."
)


def _group_sources_by_document(
    relevant_docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group retrieved chunks by source document.

    Multiple chunks from the same document are collapsed into one
    source entry with the highest relevance score among them.

    De-duplication keys by **filename** so that the same file ingested
    multiple times (different ``document_id`` values) still appears
    only once in the UI source list.

    Args:
        relevant_docs: List of retrieved document chunks with metadata.

    Returns:
        De-duplicated sources keyed by filename, highest score wins.
    """
    doc_map: dict[str, dict[str, Any]] = {}

    for doc in relevant_docs:
        meta = doc.get("metadata", {})
        filename = meta.get("filename", "")
        document_id = meta.get("document_id", "")
        # Key by filename so duplicate ingestions of the same file
        # are collapsed into a single source citation.
        key = filename or document_id or doc.get("id", str(uuid.uuid4()))

        score = doc.get("score", 0)

        if key not in doc_map or score > doc_map[key]["score"]:
            doc_map[key] = {
                "filename": filename or "Unknown source",
                "document_id": document_id,
                "score": score,
                "id": key,
            }

    return list(doc_map.values())


class ChatService:
    """Orchestrates chat interactions through the RAG pipeline.

    This service bridges the API layer with the RAG pipeline,
    managing conversation state and coordinating all dependencies.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        session_factory: Any = None,
    ) -> None:
        self._llm_provider = llm_provider
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        
        if session_factory is None:
            from app.infrastructure.database.connection import AsyncSessionLocal
            self._session_factory = AsyncSessionLocal
        else:
            self._session_factory = session_factory
        self._output_validator = OutputValidator()
        self._content_moderator = ContentModerator()
        self._escalation_detector = EscalationDetector()

    async def _is_conversation_escalated(self, conversation_id: str) -> bool:
        """Check if a conversation has already been escalated."""
        logger.debug("checking_escalation_status", conversation_id=conversation_id)
        try:
            from app.infrastructure.database.repositories.conversation_repo import (
                SQLConversationRepository,
            )

            async with self._session_factory() as session:
                repo = SQLConversationRepository(session)
                conv = await repo.get_by_id(conversation_id)
                if conv and conv.status and conv.status.value == "escalated":
                    return True
        except Exception:
            logger.debug("escalation_check_failed", exc_info=True)
        return False

    def _resolve_effective_provider(
        self,
        tenant_chat_provider: str | None,
        tenant_gemini_api_key: str | None,
        tenant_chat_model: str | None,
    ) -> tuple[LLMProvider, bool]:
        """Resolve the LLM provider for a specific request.

        If the tenant has configured Gemini with a valid API key, returns
        a per-request GeminiAdapter. Otherwise returns the default
        (Ollama) provider.

        Args:
            tenant_chat_provider: Provider identifier ("gemini" or "ollama").
            tenant_gemini_api_key: Decrypted Gemini API key.
            tenant_chat_model: Model identifier for the provider.

        Returns:
            Tuple of (provider, is_disposable).  When ``is_disposable``
            is True the caller must close the provider after use to
            avoid leaking the underlying HTTP client.
        """
        if tenant_chat_provider == "gemini" and tenant_gemini_api_key:
            from app.infrastructure.llm.factory import get_gemini_provider
            resolved_model = tenant_chat_model or "gemini-2.5-flash"
            logger.info(
                "provider_resolved",
                provider="gemini",
                model=resolved_model,
                disposable=True,
            )
            return get_gemini_provider(
                api_key=tenant_gemini_api_key,
                model=resolved_model,
            ), True
        logger.info(
            "provider_resolved",
            provider="ollama",
            model=tenant_chat_model or getattr(self._llm_provider, 'default_model', 'unknown'),
            disposable=False,
        )
        return self._llm_provider, False

    @staticmethod
    async def _close_if_disposable(
        provider: LLMProvider, disposable: bool,
    ) -> None:
        """Close a per-request provider if it was created for this call."""
        if disposable and hasattr(provider, "close"):
            try:
                await provider.close()
                logger.info(
                    "provider_closed",
                    provider=getattr(provider, 'provider_name', 'unknown'),
                )
            except Exception:  # pragma: no cover — best-effort cleanup
                logger.warning(
                    "provider_close_failed",
                    provider=getattr(provider, 'provider_name', 'unknown'),
                )
                pass

    def _resolve_effective_embedding_service(
        self,
        tenant_embedding_provider: str | None,
        tenant_gemini_embedding_api_key: str | None,
        tenant_embedding_model: str | None,
    ) -> tuple[Any, bool]:
        """Resolve the embedding provider for a specific request.

        If the tenant has Gemini embeddings configured with a valid API
        key, returns a per-request ``GeminiEmbeddingAdapter``.  Otherwise
        returns the shared Ollama ``EmbeddingService``.

        Args:
            tenant_embedding_provider: Provider identifier ("gemini" or "ollama").
            tenant_gemini_embedding_api_key: Decrypted Gemini API key for embeddings.
            tenant_embedding_model: Embedding model identifier.

        Returns:
            Tuple of (embedding_service, is_disposable).
        """
        if tenant_embedding_provider == "gemini" and tenant_gemini_embedding_api_key:
            from app.infrastructure.llm.factory import get_gemini_embedding_provider
            resolved_model = tenant_embedding_model or "gemini-embedding-2"
            logger.info(
                "embedding_provider_resolved",
                provider="gemini",
                model=resolved_model,
                disposable=True,
            )
            return get_gemini_embedding_provider(
                api_key=tenant_gemini_embedding_api_key,
                model=resolved_model,
            ), True
        if tenant_embedding_provider == "gemini" and not tenant_gemini_embedding_api_key:
            logger.warning(
                "gemini_embedding_key_missing",
                configured_provider="gemini",
                falling_back_to="ollama",
                hint="Tenant has Gemini embeddings configured but API key is missing or decryption failed",
            )
        logger.info(
            "embedding_provider_resolved",
            provider="ollama",
            model=tenant_embedding_model or "default",
            disposable=False,
        )
        return self._embedding_service, False

    async def process_message(
        self,
        message: str,
        tenant_id: str,
        conversation_id: str | None = None,
        tenant_blocklist: list[str] | None = None,
        user_id: str = "",
        tenant_chat_model: str | None = None,
        tenant_embedding_model: str | None = None,
        tenant_chat_provider: str | None = None,
        tenant_gemini_api_key: str | None = None,
        tenant_embedding_provider: str | None = None,
        tenant_gemini_embedding_api_key: str | None = None,
        tenant_agent_config: dict[str, Any] | None = None,
        tenant_config_json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process a user message through the RAG pipeline.

        Args:
            message: User's message text.
            tenant_id: Tenant context.
            conversation_id: Optional existing conversation ID.
            tenant_blocklist: Tenant-specific list of banned terms for
                content moderation. Loaded from tenant ``config_json``.
            user_id: Authenticated user's ID (for conversation persistence).
            tenant_chat_model: Tenant-specific chat model override.
            tenant_embedding_model: Tenant-specific embedding model override.
            tenant_chat_provider: Provider identifier (``"gemini"`` or
                ``"ollama"``).  If ``None``, defaults to Ollama.
            tenant_gemini_api_key: Decrypted Gemini API key for runtime
                use.  If ``None``, Gemini cannot be used.
            tenant_embedding_provider: Embedding provider identifier
                (``"gemini"`` or ``"ollama"``).  If ``None``, defaults
                to Ollama.
            tenant_gemini_embedding_api_key: Decrypted Gemini API key
                for embedding requests.  If ``None``, Gemini embeddings
                cannot be used.
            tenant_agent_config: Tenant's agent personality config from
                ``config_json["agent_prompt"]``.  Supports custom prompts,
                structured overrides (agent_name, tone, domain_rules),
                or ``None`` for the default system prompt.

        Returns:
            Dict with answer, sources, escalation status, etc.
        """
        logger.debug("chat_service_process_message_started", tenant_id=tenant_id, conversation_id=conversation_id, user_id=user_id)
        # Generate conversation ID if not provided
        is_new_conversation = not conversation_id
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        blocklist = tenant_blocklist or []

        # ── Event hook: new conversation ───────────────────────────
        if is_new_conversation:
            dispatch_event(
                tenant_config_json,
                EventType.ON_NEW_CONVERSATION,
                HookPayload(
                    event=EventType.ON_NEW_CONVERSATION.value,
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    data={"user_id": user_id},
                ),
            )

        logger.info(
            "chat_process_message",
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            message_length=len(message),
        )

        # ── Already-escalated short circuit ────────────────────────
        if not is_new_conversation and await self._is_conversation_escalated(
            conversation_id,
        ):
            logger.info(
                "post_escalation_message",
                conversation_id=conversation_id,
            )
            await self._persist_exchange(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                user_id=user_id,
                user_message=message,
                assistant_message=_POST_ESCALATION_REPLY,
                assistant_thinking="",
                sources=[],
                model_used="",
                is_new=False,
            )
            return {
                "answer": _POST_ESCALATION_REPLY,
                "conversation_id": conversation_id,
                "sources": [],
                "escalated": True,
                "escalation_reason": "Conversation already escalated",
                "escalation_trigger": "none",
                "model_used": "",
            }

        # ── Input content moderation (before RAG) ────────────────
        input_check = self._content_moderator.check_input(message, blocklist)
        if input_check.blocked:
            logger.warning(
                "content_moderation_input_blocked",
                conversation_id=conversation_id,
                reason=input_check.reason,
                matched_term=input_check.matched_term[:100],
            )

            # Persist blocked exchange for admin audit trail
            await self._persist_exchange(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                user_id=user_id,
                user_message=message,
                assistant_message=input_check.canned_response,
                assistant_thinking="",
                sources=[],
                model_used="",
                is_new=is_new_conversation,
                validation_status=ValidationStatus.FLAGGED.value,
                moderation_reason=input_check.reason,
                moderation_matched_term=input_check.matched_term,
            )

            return {
                "answer": input_check.canned_response,
                "conversation_id": conversation_id,
                "sources": [],
                "escalated": False,
                "escalation_reason": "",
                "model_used": "",
                "moderation_blocked": True,
                "moderation_reason": input_check.reason,
            }

        # Resolve effective LLM provider for this request
        effective_provider, _disposable = self._resolve_effective_provider(
            tenant_chat_provider, tenant_gemini_api_key, tenant_chat_model,
        )

        try:
            # ── Smart escalation detection (before RAG) ──────────────
            history_messages = await self._load_conversation_history(
                tenant_id, conversation_id, is_new_conversation, chat_model=tenant_chat_model,
                llm_provider=effective_provider,
            )
            escalation_check = self._escalation_detector.detect(
                message=message,
                conversation_history=history_messages,
            )
            if escalation_check.should_escalate:
                trigger = escalation_check.trigger
                answer_text = _ESCALATION_MESSAGES.get(
                    trigger, _ESCALATION_MESSAGES[EscalationTrigger.NO_CONTEXT],
                )
                logger.info(
                    "smart_escalation_triggered",
                    conversation_id=conversation_id,
                    trigger=trigger.value,
                    reason=escalation_check.reason,
                    sentiment_score=escalation_check.sentiment_score,
                    repetition_count=escalation_check.repetition_count,
                )

                await self._persist_exchange(
                    conversation_id=conversation_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    user_message=message,
                    assistant_message=answer_text,
                    assistant_thinking="",
                    sources=[],
                    model_used="",
                    is_new=is_new_conversation,
                    escalation_trigger=trigger.value,
                )

                return {
                    "answer": answer_text,
                    "conversation_id": conversation_id,
                    "sources": [],
                    "escalated": True,
                    "escalation_reason": escalation_check.reason,
                    "escalation_trigger": trigger.value,
                    "model_used": "",
                }

            # Resolve effective embedding service for this request
            effective_embed, embed_disposable = self._resolve_effective_embedding_service(
                tenant_embedding_provider, tenant_gemini_embedding_api_key,
                tenant_embedding_model,
            )
            try:
                # Run the RAG pipeline (retrieve + grade only)
                result = await run_rag_pipeline(
                    query=message,
                    tenant_id=tenant_id,
                    vector_store=self._vector_store,
                    embedding_service=effective_embed,
                    llm_provider=effective_provider,
                    chat_model=tenant_chat_model,
                    embedding_model=tenant_embedding_model,
                )
            finally:
                if embed_disposable and hasattr(effective_embed, 'close'):
                    try:
                        await effective_embed.close()
                    except Exception:  # noqa: S110
                        logger.debug("embed_adapter_close_failed", exc_info=True)

            # ── Tool loop (always runs — even for pure RAG with escalate-only) ──
            # Load encrypted secrets for tool auth
            tenant_secrets: dict[str, str] = {}
            if tenant_config_json and tenant_config_json.get("tools_enabled"):
                try:
                    from app.config import get_settings
                    from app.infrastructure.database.repositories.tenant_secret_repo import (
                        SQLTenantSecretRepository,
                    )

                    async with self._session_factory() as sec_session:
                        sec_repo = SQLTenantSecretRepository(
                            sec_session,
                            encryption_key=get_settings().secret_key,
                        )
                        tenant_secrets = await sec_repo.get_all_decrypted(tenant_id)
                except Exception:
                    logger.warning("tenant_secrets_load_failed", tenant_id=tenant_id, exc_info=True)

            tenant_tools = resolve_tenant_tools(tenant_config_json, secrets=tenant_secrets)
            max_rounds = (
                tenant_config_json.get("max_tool_rounds", 3)
                if tenant_config_json else 3
            )
            system_prompt = build_system_prompt(
                agent_config=tenant_agent_config,
                available_tools=tenant_tools,
            )
            executor = ToolExecutor(max_rounds=max_rounds)
            tool_loop_gen = run_tool_loop(
                result, tenant_tools, effective_provider, executor,
                system_prompt=system_prompt,
                conversation_history=history_messages,
                chat_model=tenant_chat_model,
            )
            async for frame in tool_loop_gen:
                if frame["type"] == "state":
                    result = frame["data"]

            # ── Event hook: tool failures ──────────────────────────
            for tr in result.get("tool_results", []):
                if not tr.get("success"):
                    dispatch_event(
                        tenant_config_json,
                        EventType.ON_TOOL_FAILURE,
                        HookPayload(
                            event=EventType.ON_TOOL_FAILURE.value,
                            tenant_id=tenant_id,
                            conversation_id=conversation_id,
                            data={
                                "tool_name": tr.get("name", ""),
                                "error": tr.get("error", ""),
                            },
                        ),
                    )

            # Generate or escalate
            if result.get("should_escalate"):
                result = await escalation_node(result)
            elif result.get("tool_answer"):
                # Tool loop already produced an answer incorporating tool results.
                # Use it directly — calling generate_node would lose tool context.
                result["answer"] = result["tool_answer"]
                result["model_used"] = tenant_chat_model or effective_provider.default_model
                result["sources"] = [
                    {
                        "content": doc.get("content", "")[:200],
                        "score": doc.get("score", 0),
                        "id": doc.get("id", ""),
                    }
                    for doc in result.get("relevant_docs", [])
                ]
            else:
                result = await generate_node(
                    result, effective_provider,
                    chat_model=tenant_chat_model,
                    history_messages=history_messages,
                    agent_config=tenant_agent_config,
                )

            # Group sources by document (de-duplicate chunks from same file)
            grouped_sources = _group_sources_by_document(result.get("relevant_docs", []))
            answer = result.get("answer", "")

            # ── Strip sentinel token anywhere in the response ─────────
            if _ESCALATE_SENTINEL in answer:
                answer = answer.replace(_ESCALATE_SENTINEL, "").strip()
                result["should_escalate"] = True
                logger.info(
                    "llm_escalation_stripped",
                    conversation_id=conversation_id,
                    reason="Sentinel token found embedded in response — stripped",
                )

            # ── Output content moderation ────────────────────────────
            output_check = self._content_moderator.check_output(answer, blocklist)
            if output_check.flagged:
                logger.warning(
                    "content_moderation_output_flagged",
                    conversation_id=conversation_id,
                    reason=output_check.reason,
                    matched_term=output_check.matched_term[:100],
                )

            should_escalate = result.get("should_escalate", False)
            trigger_val = EscalationTrigger.NO_CONTEXT.value if should_escalate else "none"

            # ── Post-generation self-escalation detection ─────────────
            if not should_escalate:
                matched = _detect_self_escalation(answer)
                if matched:
                    should_escalate = True
                    trigger_val = EscalationTrigger.LLM_DECISION.value
                    logger.info(
                        "post_gen_escalation_detected",
                        conversation_id=conversation_id,
                        matched_pattern=matched,
                    )

            # ── Failed query logging ─────────────────────────────────
            if should_escalate:
                retrieved_docs = result.get("retrieved_docs", [])
                await self._persist_failed_query(
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    query_text=message,
                    failure_reason=FailureReason.NO_DOCS,
                    retrieved_doc_count=len(retrieved_docs),
                    max_relevance_score=max(
                        (d.get("score", 0) for d in retrieved_docs), default=0.0,
                    ),
                    escalation_trigger=EscalationTrigger.NO_CONTEXT,
                )

                # ── Event hook: escalation ────────────────────────
                dispatch_event(
                    tenant_config_json,
                    EventType.ON_ESCALATION,
                    HookPayload(
                        event=EventType.ON_ESCALATION.value,
                        tenant_id=tenant_id,
                        conversation_id=conversation_id,
                        data={
                            "trigger": trigger_val,
                            "reason": result.get("escalation_reason", ""),
                        },
                    ),
                )

            return {
                "answer": answer,
                "conversation_id": conversation_id,
                "sources": grouped_sources,
                "escalated": should_escalate,
                "escalation_reason": result.get("escalation_reason", ""),
                "escalation_trigger": trigger_val,
                "model_used": result.get("model_used", ""),
            }
        finally:
            await self._close_if_disposable(effective_provider, _disposable)

    async def stream_message(
        self,
        message: str,
        tenant_id: str,
        user_id: str = "",
        conversation_id: str | None = None,
        temperature: float = 0.2,
        tenant_blocklist: list[str] | None = None,
        tenant_chat_model: str | None = None,
        tenant_embedding_model: str | None = None,
        tenant_chat_provider: str | None = None,
        tenant_gemini_api_key: str | None = None,
        tenant_embedding_provider: str | None = None,
        tenant_gemini_embedding_api_key: str | None = None,
        tenant_agent_config: dict[str, Any] | None = None,
        tenant_config_json: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream a chat response token-by-token via the RAG pipeline.

        Runs retrieval and grading synchronously, then streams the
        LLM generation step as individual token frames. Persists the
        conversation and messages to the database on completion.

        Frame types:
            - ``{"type": "token", "data": "partial text"}``
            - ``{"type": "source", "data": {"filename": ..., "score": ..., ...}}``
            - ``{"type": "done", "data": {"conversation_id": ..., ...}}``

        Args:
            message: User's message text.
            tenant_id: Tenant context.
            user_id: Authenticated user's ID (for conversation persistence).
            conversation_id: Optional existing conversation ID.
            temperature: LLM sampling temperature (0.0–1.0). Values outside
                this range are clamped to the default.
            tenant_blocklist: Tenant-specific list of banned terms for
                content moderation. Loaded from tenant ``config_json``.
            tenant_chat_model: Tenant-specific chat model override.
                If ``None``, the server's default model is used.
            tenant_embedding_model: Tenant-specific embedding model override.
                If ``None``, the server's default model is used.
            tenant_chat_provider: Provider identifier (``"gemini"`` or
                ``"ollama"``).  If ``None``, defaults to Ollama.
            tenant_gemini_api_key: Decrypted Gemini API key for runtime
                use.  If ``None``, Gemini cannot be used.
            tenant_embedding_provider: Embedding provider identifier
                (``"gemini"`` or ``"ollama"``).  If ``None``, defaults
                to Ollama.
            tenant_gemini_embedding_api_key: Decrypted Gemini API key
                for embedding requests.  If ``None``, Gemini embeddings
                cannot be used.
            tenant_agent_config: Tenant's agent personality config from
                ``config_json["agent_prompt"]``.  Supports custom prompts,
                structured overrides (agent_name, tone, domain_rules),
                or ``None`` for the default system prompt.

        Yields:
            Structured frame dicts for WebSocket delivery.
        """
        logger.debug("chat_service_stream_message_started", tenant_id=tenant_id, conversation_id=conversation_id, user_id=user_id)
        # Clamp temperature to valid range
        if not isinstance(temperature, (int, float)) or temperature < 0.0 or temperature > 1.0:
            temperature = 0.2

        # Default blocklist to empty list
        blocklist = tenant_blocklist or []

        is_new_conversation = not conversation_id
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        # ── Event hook: new conversation ───────────────────────────
        if is_new_conversation:
            dispatch_event(
                tenant_config_json,
                EventType.ON_NEW_CONVERSATION,
                HookPayload(
                    event=EventType.ON_NEW_CONVERSATION.value,
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    data={"user_id": user_id},
                ),
            )

        logger.info(
            "chat_stream_message",
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            message_length=len(message),
        )

        # ── Already-escalated short circuit ────────────────────────
        if not is_new_conversation and await self._is_conversation_escalated(
            conversation_id,
        ):
            logger.info(
                "post_escalation_message",
                conversation_id=conversation_id,
            )
            await self._persist_exchange(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                user_id=user_id,
                user_message=message,
                assistant_message=_POST_ESCALATION_REPLY,
                assistant_thinking="",
                sources=[],
                model_used="",
                is_new=False,
            )
            yield {"type": "token", "data": _POST_ESCALATION_REPLY}
            yield {
                "type": "done",
                "data": {
                    "conversation_id": conversation_id,
                    "model_used": "",
                    "sources": [],
                    "escalated": True,
                    "escalation_reason": "Conversation already escalated",
                    "escalation_trigger": "none",
                },
            }
            return

        # ── Input content moderation (before RAG) ────────────────
        input_check = self._content_moderator.check_input(message, blocklist)
        if input_check.blocked:
            logger.warning(
                "content_moderation_input_blocked",
                conversation_id=conversation_id,
                reason=input_check.reason,
                matched_term=input_check.matched_term[:100],
            )
            yield {
                "type": "token",
                "data": input_check.canned_response,
            }
            yield {
                "type": "done",
                "data": {
                    "conversation_id": conversation_id,
                    "model_used": "",
                    "sources": [],
                    "escalated": False,
                    "escalation_reason": "",
                    "moderation_blocked": True,
                    "moderation_reason": input_check.reason,
                },
            }

            # Persist the blocked exchange
            await self._persist_exchange(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                user_id=user_id,
                user_message=message,
                assistant_message=input_check.canned_response,
                assistant_thinking="",
                sources=[],
                model_used="",
                is_new=is_new_conversation,
                validation_status=ValidationStatus.FLAGGED.value,
                moderation_reason=input_check.reason,
                moderation_matched_term=input_check.matched_term,
            )
            return

        # Resolve effective LLM provider for this request
        effective_provider, _disposable = self._resolve_effective_provider(
            tenant_chat_provider, tenant_gemini_api_key, tenant_chat_model,
        )

        try:
            # ── Smart escalation detection (before RAG) ──────────────
            # Load conversation history for repetition detection
            history_messages = await self._load_conversation_history(
                tenant_id, conversation_id, is_new_conversation, chat_model=tenant_chat_model,
                llm_provider=effective_provider,
            )

            escalation_check = self._escalation_detector.detect(
                message=message,
                conversation_history=history_messages,
            )
            if escalation_check.should_escalate:
                trigger = escalation_check.trigger
                answer_text = _ESCALATION_MESSAGES.get(
                    trigger, _ESCALATION_MESSAGES[EscalationTrigger.NO_CONTEXT],
                )
                logger.info(
                    "smart_escalation_triggered",
                    conversation_id=conversation_id,
                    trigger=trigger.value,
                    reason=escalation_check.reason,
                    sentiment_score=escalation_check.sentiment_score,
                    repetition_count=escalation_check.repetition_count,
                )
                yield {
                    "type": "token",
                    "data": answer_text,
                }
                yield {
                    "type": "done",
                    "data": {
                        "conversation_id": conversation_id,
                        "model_used": "",
                        "sources": [],
                        "escalated": True,
                        "escalation_reason": escalation_check.reason,
                        "escalation_trigger": trigger.value,
                    },
                }

                await self._persist_exchange(
                    conversation_id=conversation_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    user_message=message,
                    assistant_message=answer_text,
                    assistant_thinking="",
                    sources=[],
                    model_used="",
                    is_new=is_new_conversation,
                    escalation_trigger=trigger.value,
                )

                # ── Event hook: smart escalation ─────────────────
                dispatch_event(
                    tenant_config_json,
                    EventType.ON_ESCALATION,
                    HookPayload(
                        event=EventType.ON_ESCALATION.value,
                        tenant_id=tenant_id,
                        conversation_id=conversation_id,
                        data={
                            "trigger": trigger.value,
                            "reason": escalation_check.reason,
                        },
                    ),
                )
                return

            # Step 1: Retrieve + Grade via LangGraph (non-streaming)
            state: RAGState = {
                "query": message,
                "tenant_id": tenant_id,
                "retrieved_docs": [],
                "relevant_docs": [],
                "answer": "",
                "sources": [],
                "should_escalate": False,
                "escalation_reason": "",
                "model_used": "",
                "tokens_in": 0,
                "tokens_out": 0,
                "tool_calls": [],
                "tool_results": [],
                "tool_round": 0,
                "tool_messages": [],
            }

            # Resolve effective embedding service for this request
            effective_embed, embed_disposable = self._resolve_effective_embedding_service(
                tenant_embedding_provider, tenant_gemini_embedding_api_key,
                tenant_embedding_model,
            )
            try:
                compiled = build_rag_graph(
                    self._vector_store,
                    effective_embed,
                    effective_provider,
                    embedding_model=tenant_embedding_model,
                )
                state = await compiled.ainvoke(state)
            finally:
                if embed_disposable and hasattr(effective_embed, 'close'):
                    try:
                        await effective_embed.close()
                    except Exception:  # noqa: S110
                        logger.debug("embed_adapter_close_failed", exc_info=True)

            # Step 2: Check RAG-level escalation (no relevant docs found)
            # Only trigger this premature escalation if tools are disabled.
            # If tools are enabled, the LLM might be able to handle the query via a tool call
            # even without retrieved documents.
            tools_enabled = tenant_config_json.get("tools_enabled", False) if tenant_config_json else False
            
            if state.get("should_escalate"):
                if tools_enabled:
                    # Let the tool loop attempt to handle it even without RAG context
                    state["should_escalate"] = False
                    state["escalation_reason"] = ""
                else:
                    rag_reason = state.get("escalation_reason", "")
                    answer_text = _ESCALATION_MESSAGES[EscalationTrigger.NO_CONTEXT]
                    logger.info(
                        "rag_escalation_triggered",
                        conversation_id=conversation_id,
                        reason=rag_reason,
                    )
                    yield {
                        "type": "token",
                        "data": answer_text,
                    }
                    yield {
                        "type": "done",
                        "data": {
                            "conversation_id": conversation_id,
                            "model_used": "",
                            "sources": [],
                            "escalated": True,
                            "escalation_reason": rag_reason,
                            "escalation_trigger": EscalationTrigger.NO_CONTEXT.value,
                        },
                    }
    
                    # Persist escalation to DB
                    await self._persist_exchange(
                        conversation_id=conversation_id,
                        tenant_id=tenant_id,
                        user_id=user_id,
                        user_message=message,
                        assistant_message=answer_text,
                        assistant_thinking="",
                        sources=[],
                        model_used="",
                        is_new=is_new_conversation,
                        escalation_trigger=EscalationTrigger.NO_CONTEXT.value,
                    )
    
                    # ── Failed query logging ─────────────────────────────
                    retrieved_docs = state.get("retrieved_docs", [])
                    await self._persist_failed_query(
                        tenant_id=tenant_id,
                        conversation_id=conversation_id,
                        query_text=message,
                        failure_reason=FailureReason.NO_DOCS,
                        retrieved_doc_count=len(retrieved_docs),
                        max_relevance_score=max(
                            (d.get("score", 0) for d in retrieved_docs), default=0.0,
                        ),
                        escalation_trigger=EscalationTrigger.NO_CONTEXT,
                    )
    
                    # ── Event hook: RAG-level escalation ──────────────
                    dispatch_event(
                        tenant_config_json,
                        EventType.ON_ESCALATION,
                        HookPayload(
                            event=EventType.ON_ESCALATION.value,
                            tenant_id=tenant_id,
                            conversation_id=conversation_id,
                            data={
                                "trigger": EscalationTrigger.NO_CONTEXT.value,
                                "reason": "No relevant documents found",
                            },
                        ),
                    )
                    return

            # Step 3: Run tool loop (always runs — even for pure RAG)
            # Load encrypted secrets for tool auth
            tenant_secrets: dict[str, str] = {}
            if tenant_config_json and tenant_config_json.get("tools_enabled"):
                try:
                    from app.config import get_settings
                    from app.infrastructure.database.repositories.tenant_secret_repo import (
                        SQLTenantSecretRepository,
                    )

                    async with self._session_factory() as sec_session:
                        sec_repo = SQLTenantSecretRepository(
                            sec_session,
                            encryption_key=get_settings().secret_key,
                        )
                        tenant_secrets = await sec_repo.get_all_decrypted(tenant_id)
                except Exception:
                    logger.warning("tenant_secrets_load_failed", tenant_id=tenant_id, exc_info=True)

            tenant_tools = resolve_tenant_tools(tenant_config_json, secrets=tenant_secrets)
            max_rounds = (
                tenant_config_json.get("max_tool_rounds", 3)
                if tenant_config_json else 3
            )
            system_prompt = build_system_prompt(
                agent_config=tenant_agent_config,
                available_tools=tenant_tools,
            )
            executor = ToolExecutor(max_rounds=max_rounds)
            tool_loop_gen = run_tool_loop(
                state, tenant_tools, effective_provider, executor,
                system_prompt=system_prompt,
                conversation_history=history_messages,
                chat_model=tenant_chat_model,
            )
            async for frame in tool_loop_gen:
                if frame["type"] == "state":
                    state = frame["data"]
                else:
                    yield frame

            # ── Event hook: tool failures ──────────────────────────
            for tr in state.get("tool_results", []):
                if not tr.get("success"):
                    dispatch_event(
                        tenant_config_json,
                        EventType.ON_TOOL_FAILURE,
                        HookPayload(
                            event=EventType.ON_TOOL_FAILURE.value,
                            tenant_id=tenant_id,
                            conversation_id=conversation_id,
                            data={
                                "tool_name": tr.get("name", ""),
                                "error": tr.get("error", ""),
                            },
                        ),
                    )

            # Check if tool loop triggered escalation
            if state.get("should_escalate"):
                tool_reason = state.get("escalation_reason", "Tool-triggered escalation")
                answer_text = _ESCALATION_MESSAGES[EscalationTrigger.LLM_DECISION]
                logger.info(
                    "tool_escalation_triggered",
                    conversation_id=conversation_id,
                    reason=tool_reason,
                )
                yield {"type": "token", "data": answer_text}
                yield {
                    "type": "done",
                    "data": {
                        "conversation_id": conversation_id,
                        "model_used": "",
                        "sources": [],
                        "escalated": True,
                        "escalation_reason": tool_reason,
                        "escalation_trigger": EscalationTrigger.LLM_DECISION.value,
                    },
                }
                await self._persist_exchange(
                    conversation_id=conversation_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    user_message=message,
                    assistant_message=answer_text,
                    assistant_thinking="",
                    sources=[],
                    model_used="",
                    is_new=is_new_conversation,
                    escalation_trigger=EscalationTrigger.LLM_DECISION.value,
                )

                # ── Event hook: tool-triggered escalation ────────
                dispatch_event(
                    tenant_config_json,
                    EventType.ON_ESCALATION,
                    HookPayload(
                        event=EventType.ON_ESCALATION.value,
                        tenant_id=tenant_id,
                        conversation_id=conversation_id,
                        data={
                            "trigger": EscalationTrigger.LLM_DECISION.value,
                            "reason": tool_reason,
                        },
                    ),
                )
                return

            # Step 4: Build context and stream generation
            relevant_docs = state.get("relevant_docs", [])

            # Group sources by document for the UI
            grouped_sources = _group_sources_by_document(relevant_docs)

            # If tool loop already produced an answer, yield it directly
            # instead of streaming from the LLM (which would lack tool context).
            if state.get("tool_answer"):
                tool_answer_text = state["tool_answer"]
                for source in grouped_sources:
                    yield {"type": "source", "data": source}
                yield {"type": "token", "data": tool_answer_text}
                model_used = tenant_chat_model or getattr(effective_provider, "default_model", "")

                yield {
                    "type": "done",
                    "data": {
                        "conversation_id": conversation_id,
                        "model_used": model_used,
                        "sources": grouped_sources,
                        "escalated": False,
                        "retrieval_scores": {
                            "max_score": max(
                                (d.get("score", 0) for d in relevant_docs), default=0.0,
                            ),
                        },
                    },
                }
                await self._persist_exchange(
                    conversation_id=conversation_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    user_message=message,
                    assistant_message=tool_answer_text,
                    assistant_thinking="",
                    sources=grouped_sources,
                    model_used=model_used,
                    is_new=is_new_conversation,
                )
                return

            # Unified prompt + message construction (shared with generate_node)
            context = format_rag_context(relevant_docs)
            messages = build_rag_messages(
                query=message,
                context=context,
                history_messages=history_messages,
                agent_config=tenant_agent_config,
                available_tools=tenant_tools,
            )

            # Yield grouped source citations before streaming tokens

            for source in grouped_sources:
                yield {"type": "source", "data": source}

            # ── Stream LLM tokens ────────────────────────────────────
            full_answer_parts: list[str] = []
            full_thinking_parts: list[str] = []
            llm_escalated = False

            # Buffer early content tokens to detect [ESCALATE] sentinel
            # before any content reaches the client.
            _sentinel_len = len(_ESCALATE_SENTINEL)
            _content_buffer: list[str] = []
            _buffer_flushed = False
            _sentinel_tail = ""  # Rolling buffer for late [ESCALATE] filtering

            # DEBUG: Dump the full messages payload being sent to LLM
            provider_name = getattr(effective_provider, 'provider_name', 'unknown')
            logger.info(
                "llm_request_start",
                conversation_id=conversation_id,
                provider=provider_name,
                model=tenant_chat_model or getattr(effective_provider, 'default_model', ''),
                message_count=len(messages),
                temperature=temperature,
            )
            for idx, msg in enumerate(messages):
                content = msg["content"]
                logger.debug(
                    "llm_message_payload",
                    conversation_id=conversation_id,
                    provider=provider_name,
                    index=idx,
                    role=msg["role"],
                    content_length=len(content),
                    content_head=content[:300],
                    content_tail=content[-400:] if len(content) > 400 else "(shown in head)",
                )

            async for token_frame in effective_provider.stream(  # type: ignore[attr-defined]
                messages=messages, model=tenant_chat_model, temperature=temperature,
            ):
                frame_kind = token_frame.get("type", "content") if isinstance(token_frame, dict) else "content"
                token_text = token_frame.get("text", "") if isinstance(token_frame, dict) else token_frame

                if frame_kind == "thinking":
                    full_thinking_parts.append(token_text)
                    yield {"type": "thinking", "data": token_text}
                    continue

                # Content token — check for sentinel in early tokens
                if not _buffer_flushed:
                    _content_buffer.append(token_text)
                    buffered = "".join(_content_buffer).lstrip()

                    # Enough chars to decide?
                    if len(buffered) >= _sentinel_len:
                        if buffered.startswith(_ESCALATE_SENTINEL):
                            # Escalation detected — discard all LLM output
                            llm_escalated = True
                            full_answer_parts = [_ESCALATION_MESSAGES[EscalationTrigger.LLM_DECISION]]
                            yield {"type": "token", "data": full_answer_parts[0]}
                            logger.info(
                                "llm_escalation_triggered",
                                conversation_id=conversation_id,
                                reason="LLM decided to escalate via sentinel token",
                            )
                            # Stop consuming further tokens
                            break
                        else:
                            # No sentinel — flush buffer as normal tokens
                            _buffer_flushed = True
                            for buf_token in _content_buffer:
                                full_answer_parts.append(buf_token)
                                yield {"type": "token", "data": buf_token}
                else:
                    # ── Late-sentinel filter ──────────────────────────
                    # Buffer tokens when we see '[' to catch [ESCALATE]
                    # that appears mid- or end-of-response.  When no '['
                    # is pending, tokens pass through with zero latency.
                    _sentinel_tail += token_text

                    if "[" not in _sentinel_tail:
                        # Fast path — no bracket, yield immediately
                        full_answer_parts.append(_sentinel_tail)
                        yield {"type": "token", "data": _sentinel_tail}
                        _sentinel_tail = ""
                        continue

                    bracket_idx = _sentinel_tail.find("[")
                    tail_from_bracket = _sentinel_tail[bracket_idx:]

                    if tail_from_bracket == _ESCALATE_SENTINEL:
                        # Exact match — swallow the sentinel
                        llm_escalated = True
                        safe = _sentinel_tail[:bracket_idx]
                        if safe:
                            full_answer_parts.append(safe)
                            yield {"type": "token", "data": safe}
                        _sentinel_tail = ""
                    elif _ESCALATE_SENTINEL.startswith(tail_from_bracket):
                        # Partial match — keep buffering, yield safe prefix
                        safe = _sentinel_tail[:bracket_idx]
                        if safe:
                            full_answer_parts.append(safe)
                            yield {"type": "token", "data": safe}
                        _sentinel_tail = tail_from_bracket
                    elif _ESCALATE_SENTINEL in _sentinel_tail:
                        # Match with surrounding text — strip it
                        llm_escalated = True
                        cleaned = _sentinel_tail.replace(_ESCALATE_SENTINEL, "")
                        if cleaned:
                            full_answer_parts.append(cleaned)
                            yield {"type": "token", "data": cleaned}
                        _sentinel_tail = ""
                    else:
                        # '[' but not sentinel — flush everything
                        full_answer_parts.append(_sentinel_tail)
                        yield {"type": "token", "data": _sentinel_tail}
                        _sentinel_tail = ""

            # Flush any remaining sentinel buffer after stream ends
            if _sentinel_tail:
                if _ESCALATE_SENTINEL in _sentinel_tail:
                    llm_escalated = True
                    _sentinel_tail = _sentinel_tail.replace(_ESCALATE_SENTINEL, "")
                if _sentinel_tail.strip():
                    full_answer_parts.append(_sentinel_tail)
                    yield {"type": "token", "data": _sentinel_tail}

            # If buffer was never flushed (very short response), check and flush now
            if not _buffer_flushed and not llm_escalated:
                buffered = "".join(_content_buffer).lstrip()
                if buffered.startswith(_ESCALATE_SENTINEL):
                    llm_escalated = True
                    full_answer_parts = [_ESCALATION_MESSAGES[EscalationTrigger.LLM_DECISION]]
                    yield {"type": "token", "data": full_answer_parts[0]}
                    logger.info(
                        "llm_escalation_triggered",
                        conversation_id=conversation_id,
                        reason="LLM decided to escalate via sentinel token",
                    )
                else:
                    for buf_token in _content_buffer:
                        full_answer_parts.append(buf_token)
                        yield {"type": "token", "data": buf_token}

            model_used = tenant_chat_model or getattr(effective_provider, "default_model", "")
            full_answer = "".join(full_answer_parts)
            full_thinking = "".join(full_thinking_parts)

            # ── Strip sentinel token anywhere in the response ─────────
            # The early-buffer check catches [ESCALATE] at the start.
            # This catches it when the LLM embeds it mid- or end-of-response.
            if not llm_escalated and _ESCALATE_SENTINEL in full_answer:
                llm_escalated = True
                full_answer = full_answer.replace(_ESCALATE_SENTINEL, "").strip()
                logger.info(
                    "llm_escalation_stripped",
                    conversation_id=conversation_id,
                    reason="Sentinel token found embedded in response — stripped",
                )

            # DEBUG: Dump reasoning and answer as separate log lines
            provider_name = getattr(effective_provider, 'provider_name', 'unknown')
            logger.info(
                "llm_thinking_trace",
                conversation_id=conversation_id,
                provider=provider_name,
                model=model_used,
                thinking_length=len(full_thinking),
                thinking_preview=full_thinking[:500] if full_thinking else "(none)",
            )
            logger.info(
                "llm_final_answer",
                conversation_id=conversation_id,
                provider=provider_name,
                model=model_used,
                answer_length=len(full_answer),
                answer_preview=full_answer[:500],
                escalated=llm_escalated,
            )
            logger.info(
                "llm_stream_complete",
                conversation_id=conversation_id,
                provider=provider_name,
                model=model_used,
                thinking_chars=len(full_thinking),
                answer_chars=len(full_answer),
                escalated=llm_escalated,
            )

            # ── Post-generation output validation ────────────────────
            context_texts = [doc.get("content", "") for doc in relevant_docs]
            validation_result = self._output_validator.validate(full_answer, context_texts)
            validation_status = validation_result.status

            if validation_status == ValidationStatus.FLAGGED:
                # Append disclaimer to persisted message
                full_answer += "\n\n" + validation_result.disclaimer
                # Emit disclaimer frame so the client can display it
                yield {"type": "disclaimer", "data": validation_result.disclaimer}
                for violation in validation_result.violations:
                    logger.warning(
                        "output_validation_failed",
                        conversation_id=conversation_id,
                        rule_violated=violation.rule,
                        snippet=violation.snippet,
                    )

            # ── Post-generation content moderation (output) ──────────
            output_check = self._content_moderator.check_output(full_answer, blocklist)
            if output_check.flagged:
                # Override validation_status to flagged if not already
                validation_status = ValidationStatus.FLAGGED
                logger.warning(
                    "content_moderation_output_flagged",
                    conversation_id=conversation_id,
                    reason=output_check.reason,
                    matched_term=output_check.matched_term[:100],
                )

            # ── Post-generation self-escalation detection ─────────────
            if not llm_escalated:
                matched = _detect_self_escalation(full_answer)
                if matched:
                    llm_escalated = True
                    logger.info(
                        "post_gen_escalation_detected",
                        conversation_id=conversation_id,
                        matched_pattern=matched,
                    )

            # Resolve escalation trigger for persistence
            escalation_trigger_val = (
                EscalationTrigger.LLM_DECISION.value if llm_escalated else "none"
            )

            # Persist to database BEFORE done frame so we can include message_id
            moderation_reason = ""
            moderation_matched = ""
            if output_check.flagged:
                moderation_reason = output_check.reason
                moderation_matched = output_check.matched_term

            assistant_message_id = await self._persist_exchange(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                user_id=user_id,
                user_message=message,
                assistant_message=full_answer,
                assistant_thinking=full_thinking,
                sources=grouped_sources,
                model_used=model_used,
                is_new=is_new_conversation,
                validation_status=validation_status.value,
                moderation_reason=moderation_reason,
                moderation_matched_term=moderation_matched,
                escalation_trigger=escalation_trigger_val,
            )

            # Done frame (includes message_id for frontend feedback)
            yield {
                "type": "done",
                "data": {
                    "conversation_id": conversation_id,
                    "message_id": assistant_message_id,
                    "model_used": model_used,
                    "sources": grouped_sources,
                    "escalated": llm_escalated,
                    "escalation_reason": "LLM determined human agent needed" if llm_escalated else "",
                    "escalation_trigger": escalation_trigger_val,
                    "thinking_text": full_thinking,
                    "validation_status": validation_status.value,
                },
            }

            # ── Failed query logging for post-gen escalation ──────────
            if llm_escalated:
                await self._persist_failed_query(
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    query_text=message,
                    failure_reason=FailureReason.NO_DOCS,
                    retrieved_doc_count=len(relevant_docs),
                    max_relevance_score=max(
                        (d.get("score", 0) for d in relevant_docs), default=0.0,
                    ),
                    escalation_trigger=EscalationTrigger.LLM_DECISION,
                    message_id=assistant_message_id or "",
                )

                # ── Event hook: post-gen LLM escalation ───────────
                dispatch_event(
                    tenant_config_json,
                    EventType.ON_ESCALATION,
                    HookPayload(
                        event=EventType.ON_ESCALATION.value,
                        tenant_id=tenant_id,
                        conversation_id=conversation_id,
                        data={
                            "trigger": EscalationTrigger.LLM_DECISION.value,
                            "reason": "LLM determined human agent needed",
                        },
                    ),
                )

        finally:
            await self._close_if_disposable(effective_provider, _disposable)

    # ── Conversation History (rolling summarization) ──────────────

    @property
    def _history_summarize_threshold(self) -> int:
        """Number of user-assistant pairs before summarization kicks in."""
        from app.config import get_settings
        return get_settings().history_summarize_threshold

    async def _load_conversation_history(
        self,
        tenant_id: str,
        conversation_id: str,
        is_new: bool,
        *,
        chat_model: str | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> list[dict[str, str]]:
        """Load recent conversation history for multi-turn context.

        Returns a list of ``{"role": ..., "content": ...}`` dicts ready
        to splice into the LLM messages array.

        Implements rolling summarization: when conversation history
        exceeds ``_HISTORY_SUMMARIZE_THRESHOLD`` pairs (user+assistant),
        older messages are compressed into a single summary message
        via an LLM call.  The summary is persisted in the DB with
        ``role=summary`` so it can be reused on subsequent turns.

        The UI never sees summary messages — they are filtered out
        in the API response layer.

        Args:
            conversation_id: UUID of the current conversation.
            is_new: If True, there is no prior history to load.

        Returns:
            List of message dicts (oldest-first) for the LLM.
        """
        if is_new:
            return []

        try:
            from app.infrastructure.database.repositories.conversation_repo import (
                SQLConversationRepository,
                SQLMessageRepository,
            )

            async with self._session_factory() as session:
                conv_repo = SQLConversationRepository(session)
                conv = await conv_repo.get_by_id(conversation_id)
                if not conv or conv.tenant_id != tenant_id:
                    logger.warning(
                        "unauthorized_conversation_access",
                        tenant_id=tenant_id,
                        conversation_id=conversation_id,
                    )
                    return []

                msg_repo = SQLMessageRepository(session)
                all_messages = await msg_repo.list_by_conversation(
                    conversation_id, limit=50,
                )

            if not all_messages:
                return []

            # Separate: find latest summary and messages after it
            latest_summary = None
            recent_messages: list = []
            for m in all_messages:
                role_val = m.role.value if hasattr(m.role, "value") else m.role
                if role_val == "summary":
                    latest_summary = m
                    recent_messages = []  # reset — only keep messages AFTER summary
                else:
                    if m.content:  # skip empty
                        recent_messages.append(m)

            # Count user-assistant pairs in recent messages
            pair_count = sum(
                1 for m in recent_messages
                if (m.role.value if hasattr(m.role, "value") else m.role) == "user"
            )

            # If we have enough pairs, trigger summarization
            if pair_count >= self._history_summarize_threshold:
                summary_text = await self._summarize_history(
                    previous_summary=latest_summary,
                    messages=recent_messages,
                    chat_model=chat_model,
                    llm_provider=llm_provider,
                )
                # Persist the new summary
                await self._persist_summary(conversation_id, summary_text)

                logger.info(
                    "history_summarized",
                    conversation_id=conversation_id,
                    pairs_summarized=pair_count,
                    summary_length=len(summary_text),
                    had_previous_summary=latest_summary is not None,
                )

                # Return only the summary for the LLM
                return [{
                    "role": "user",
                    "content": (
                        f"[Previous conversation summary — use as context, "
                        f"do not repeat this back to the customer]\n{summary_text}"
                    ),
                }]

            # Otherwise return summary (if any) + recent messages
            history: list[dict[str, str]] = []
            if latest_summary:
                history.append({
                    "role": "user",
                    "content": (
                        f"[Previous conversation summary — use as context, "
                        f"do not repeat this back to the customer]\n"
                        f"{latest_summary.content}"
                    ),
                })
            for m in recent_messages:
                role_val = m.role.value if hasattr(m.role, "value") else m.role
                history.append({"role": role_val, "content": m.content})

            return history

        except Exception:
            logger.warning(
                "chat_history_load_failed",
                conversation_id=conversation_id,
                exc_info=True,
            )
            return []  # non-critical — proceed without history

    async def _summarize_history(
        self,
        previous_summary: object | None,
        messages: list,
        *,
        chat_model: str | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> str:
        """Compress conversation history into a short summary via LLM.

        The summary preserves all customer-provided details (dates, order
        numbers, amounts) without alteration and prioritizes the most
        recent agent reply.

        Args:
            previous_summary: Existing summary message object (if rolling).
            messages: Recent user+assistant messages to summarize.
            chat_model: Optional tenant-specific chat model override.

        Returns:
            Summary text (3-N sentences, scaled by threshold).
        """
        # Build the conversation text to summarize
        parts: list[str] = []
        if previous_summary:
            content = previous_summary.content if hasattr(previous_summary, "content") else str(previous_summary)
            parts.append(f"[Previous summary]\n{content}\n")

        for m in messages:
            role_val = m.role.value if hasattr(m.role, "value") else m.role
            label = "Customer" if role_val == "user" else "Agent"
            parts.append(f"{label}: {m.content}")

        conversation_text = "\n\n".join(parts)

        # Scale sentence range based on threshold
        threshold = self._history_summarize_threshold
        max_sentences = 20 if threshold > 5 else 10

        summary_prompt = [
            {"role": "system", "content": (
                f"Summarize this customer support conversation in 3-{max_sentences} sentences.\n"
                "Rules:\n"
                "- Preserve ALL customer-provided details exactly "
                "(dates, order numbers, names, amounts, product names).\n"
                "- Include what the agent suggested or resolved. "
                "Prioritize the most recent agent reply.\n"
                "- Do NOT add information that wasn't in the conversation.\n"
                "- Output ONLY the summary, nothing else."
            )},
            {"role": "user", "content": conversation_text},
        ]

        try:
            provider = llm_provider or self._llm_provider
            summary = await provider.generate(
                messages=summary_prompt,
                model=chat_model,
                temperature=0.3,  # low temp for factual accuracy
                max_tokens=512,
            )
            return summary.strip()
        except Exception:
            logger.warning(
                "history_summarization_failed",
                exc_info=True,
            )
            # Fallback: just use the last 2 messages as a crude summary
            fallback_parts = []
            for m in messages[-2:]:
                role_val = m.role.value if hasattr(m.role, "value") else m.role
                label = "Customer" if role_val == "user" else "Agent"
                fallback_parts.append(f"{label}: {m.content}")
            return "\n".join(fallback_parts)

    async def _persist_summary(
        self,
        conversation_id: str,
        summary_text: str,
    ) -> None:
        """Save a summary message to the database.

        The summary is stored as a regular message with ``role=summary``.
        It is filtered out from API responses so the UI never sees it.

        Args:
            conversation_id: UUID of the conversation.
            summary_text: The summarized conversation text.
        """
        try:
            from app.domain.models.conversation import Message
            from app.domain.models.enums import MessageRole
            from app.infrastructure.database.repositories.conversation_repo import (
                SQLMessageRepository,
            )

            async with self._session_factory() as session:
                msg_repo = SQLMessageRepository(session)
                await msg_repo.create(
                    Message(
                        conversation_id=conversation_id,
                        role=MessageRole.SUMMARY,
                        content=summary_text,
                    )
                )
                await session.commit()
        except Exception:
            logger.warning(
                "summary_persist_failed",
                conversation_id=conversation_id,
                exc_info=True,
            )

    async def _persist_exchange(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        user_id: str,
        user_message: str,
        assistant_message: str,
        assistant_thinking: str,
        sources: list[dict[str, Any]],
        model_used: str,
        is_new: bool,
        validation_status: str = "none",
        moderation_reason: str = "",
        moderation_matched_term: str = "",
        escalation_trigger: str = "none",
    ) -> str:
        """Save the conversation and its messages to the database.

        Creates a new conversation record if ``is_new`` is True, then
        appends both the user message and the assistant response.

        This runs AFTER streaming is complete to avoid blocking the
        token-by-token delivery.

        Args:
            conversation_id: UUID of the conversation.
            tenant_id: Tenant context.
            user_id: User who initiated the chat.
            user_message: The user's original question.
            assistant_message: The full AI response.
            assistant_thinking: The full LLM reasoning trace.
            sources: Grouped source citations.
            model_used: LLM model name.
            is_new: Whether this is a brand-new conversation.
            validation_status: Output validation result (``passed``, ``flagged``, or ``none``).
            moderation_reason: Machine-readable reason for moderation action
                (e.g. ``jailbreak_detected``, ``blocklist_match``). Empty if
                no moderation was triggered.
            moderation_matched_term: The specific term or pattern that
                triggered moderation. Truncated to 200 chars. Empty if no
                moderation was triggered.
            escalation_trigger: Escalation trigger type (``none``, ``sentiment``,
                ``repetition``, ``explicit_request``, ``no_context``). Stored
                on the conversation record for analytics.

        Returns:
            The database UUID of the saved assistant message, or empty string
            on failure.
        """
        try:
            from app.domain.models.conversation import Message
            from app.domain.models.enums import (
                ConversationStatus,
                MessageRole,
            )
            from app.domain.models.enums import EscalationTrigger as ETrigger
            from app.domain.models.enums import ValidationStatus as VStatus
            from app.infrastructure.database.repositories.conversation_repo import (
                SQLConversationRepository,
                SQLMessageRepository,
            )

            # Resolve validation status enum
            try:
                vs_enum = VStatus(validation_status)
            except ValueError:
                vs_enum = VStatus.NONE

            async with self._session_factory() as session:
                conv_repo = SQLConversationRepository(session)
                msg_repo = SQLMessageRepository(session)

                # Create conversation if new
                if is_new:
                    await conv_repo.create(
                        tenant_id=tenant_id,
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )

                # Save user message
                await msg_repo.create(
                    Message(
                        conversation_id=conversation_id,
                        role=MessageRole.USER,
                        content=user_message,
                    )
                )

                # Save assistant response (use fallback if LLM returned nothing)
                saved_content = assistant_message or "(No response generated)"
                saved_msg = await msg_repo.create(
                    Message(
                        conversation_id=conversation_id,
                        role=MessageRole.ASSISTANT,
                        content=saved_content,
                        thinking=assistant_thinking,
                        sources_json=sources,
                        model_used=model_used,
                        validation_status=vs_enum,
                        moderation_reason=moderation_reason,
                        moderation_matched_term=moderation_matched_term[:200],
                    )
                )
                assistant_message_id = saved_msg.id

                await session.commit()

                # Update escalation trigger on conversation if needed
                if escalation_trigger and escalation_trigger != "none":
                    try:
                        et_enum = ETrigger(escalation_trigger)
                    except ValueError:
                        et_enum = ETrigger.NONE
                    if et_enum != ETrigger.NONE:
                        await conv_repo.update_escalation_trigger(
                            conversation_id, et_enum,
                        )
                        await conv_repo.update_status(
                            conversation_id, ConversationStatus.ESCALATED,
                        )
                        await session.commit()

            logger.info(
                "chat_exchange_persisted",
                conversation_id=conversation_id,
                is_new=is_new,
            )
            return assistant_message_id
        except Exception:
            # Log but don't fail the chat — persistence is best-effort
            logger.error(
                "chat_persist_failed",
                conversation_id=conversation_id,
                exc_info=True,
            )
            return ""

    async def _persist_failed_query(
        self,
        *,
        tenant_id: str,
        conversation_id: str,
        query_text: str,
        failure_reason: FailureReason,
        retrieved_doc_count: int = 0,
        max_relevance_score: float = 0.0,
        escalation_trigger: EscalationTrigger = EscalationTrigger.NONE,
        message_id: str = "",
    ) -> None:
        """Log a failed query for admin analytics.

        Best-effort — failures are logged but do not interrupt chat flow.

        Args:
            tenant_id: Tenant context.
            conversation_id: UUID of the conversation.
            query_text: The user's original question.
            failure_reason: Why the query failed.
            retrieved_doc_count: Number of docs retrieved from vector store.
            max_relevance_score: Highest relevance score among retrieved docs.
            escalation_trigger: What triggered the escalation.
            message_id: Optional associated message UUID.
        """
        try:
            from app.domain.models.failed_query import FailedQuery
            from app.infrastructure.database.repositories.failed_query_repo import (
                SQLFailedQueryRepository,
            )

            async with self._session_factory() as session:
                repo = SQLFailedQueryRepository(session)
                await repo.create(
                    FailedQuery(
                        tenant_id=tenant_id,
                        conversation_id=conversation_id,
                        message_id=message_id,
                        query_text=query_text,
                        failure_reason=failure_reason,
                        retrieved_doc_count=retrieved_doc_count,
                        max_relevance_score=max_relevance_score,
                        escalation_trigger=escalation_trigger,
                    )
                )
                await session.commit()

            logger.info(
                "failed_query_logged",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                failure_reason=failure_reason.value,
            )
        except Exception:
            logger.error(
                "failed_query_persist_failed",
                conversation_id=conversation_id,
                exc_info=True,
            )
