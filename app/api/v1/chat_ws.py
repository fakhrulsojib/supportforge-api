"""WebSocket chat streaming endpoint — WS /api/v1/ws/chat.

Provides real-time token-by-token streaming of RAG responses.
Supports two authentication modes:
    1. **JWT auth** — admin/agent users via supportforge-ui
    2. **Widget auth** — anonymous visitors via embeddable SDK (ws_ prefix)

JSON Frame Protocol:
    → Client sends: ``{"message": "user question", "conversation_id": "optional"}``
    ← Server sends:
        ``{"type": "source",   "data": {"content": "...", "score": 0.9, "id": "..."}}``
        ``{"type": "thinking", "data": "reasoning text"}``
        ``{"type": "token",    "data": "partial text"}``
        ``{"type": "done",     "data": {"conversation_id": "...", "thinking_text": "...", ...}}``
        ``{"type": "error",    "data": {"message": "..."}}``
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.config import get_settings
from app.core.exceptions import AuthError, LLMError, SupportForgeError
from app.core.security import verify_token
from app.core.widget_token import WIDGET_TOKEN_PREFIX, verify_widget_token
from app.domain.models.enums import TenantStatus
from app.infrastructure.database.repositories.tenant_repo import SQLTenantRepository
from app.infrastructure.database.repositories.user_repo import SQLUserRepository

if TYPE_CHECKING:
    from app.config import Settings
    from app.domain.models.tenant import Tenant

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["chat-ws"])


@dataclass
class _ResolvedConnection:
    """Holds resolved auth + tenant config for a WebSocket connection.

    Extracted into a dataclass to share config resolution logic between
    the JWT auth path and the widget auth path without duplication.
    """

    tenant_id: str
    user_id: str  # Empty string for anonymous widget visitors
    temperature: float = 0.2
    blocklist: list[str] = field(default_factory=list)
    chat_model: str | None = None
    embedding_model: str | None = None
    chat_provider: str | None = None
    gemini_api_key: str | None = None
    embedding_provider: str | None = None
    gemini_embedding_api_key: str | None = None
    agent_config: dict | None = None
    config_json: dict | None = None


def _extract_tenant_config(tenant: Tenant) -> dict:
    """Extract LLM/agent config from a tenant domain model.

    Returns a dict of config values. Shared between JWT and widget
    auth paths to avoid duplicating the config extraction logic.
    """
    result: dict = {
        "temperature": 0.2,
        "blocklist": [],
        "chat_model": None,
        "embedding_model": None,
        "chat_provider": None,
        "gemini_api_key": None,
        "embedding_provider": None,
        "gemini_embedding_api_key": None,
        "agent_config": None,
        "config_json": None,
    }

    config_json = tenant.config_json
    if not config_json:
        return result

    result["config_json"] = config_json

    raw = config_json.get("temperature")
    if isinstance(raw, (int, float)) and 0.0 <= float(raw) <= 1.0:
        result["temperature"] = float(raw)

    raw_blocklist = config_json.get("moderation_blocklist")
    if isinstance(raw_blocklist, list):
        result["blocklist"] = [str(t) for t in raw_blocklist if t]

    # Per-tenant model selections (admin-configurable)
    from app.core.tenant_config import resolve_tenant_models

    settings = get_settings()
    tenant_models = resolve_tenant_models(
        config_json,
        encryption_key=settings.secret_key,
    )
    result["chat_model"] = tenant_models.chat_model
    result["embedding_model"] = tenant_models.embedding_model
    result["chat_provider"] = tenant_models.chat_provider
    result["gemini_api_key"] = tenant_models.gemini_api_key
    result["embedding_provider"] = tenant_models.embedding_provider
    result["gemini_embedding_api_key"] = tenant_models.gemini_embedding_api_key

    # Agent personality config (no decryption — plain dict)
    raw_agent_config = config_json.get("agent_prompt")
    result["agent_config"] = raw_agent_config if isinstance(raw_agent_config, dict) else None

    return result


async def _authenticate_jwt(
    token: str,
    settings: Settings,
) -> _ResolvedConnection | None:
    """Authenticate via standard JWT token (admin/agent users).

    Returns a _ResolvedConnection on success, None on failure (caller
    should close the WebSocket with the appropriate error).
    """
    from app.infrastructure.database.connection import AsyncSessionLocal

    try:
        payload = verify_token(
            token=token,
            secret_key=settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
            expected_type="access",
        )
    except AuthError:
        return None

    async with AsyncSessionLocal() as session:
        user_repo = SQLUserRepository(session)
        user = await user_repo.get_by_id(payload.user_id)
        if not user:
            return None

        tenant_repo = SQLTenantRepository(session)
        tenant = await tenant_repo.get_by_id(user.tenant_id)

        if not tenant or tenant.status != TenantStatus.ACTIVE:
            return None

        cfg = _extract_tenant_config(tenant)

    return _ResolvedConnection(
        tenant_id=user.tenant_id,
        user_id=user.id,
        temperature=cfg["temperature"],
        blocklist=cfg["blocklist"],
        chat_model=cfg["chat_model"],
        embedding_model=cfg["embedding_model"],
        chat_provider=cfg["chat_provider"],
        gemini_api_key=cfg["gemini_api_key"],
        embedding_provider=cfg["embedding_provider"],
        gemini_embedding_api_key=cfg["gemini_embedding_api_key"],
        agent_config=cfg["agent_config"],
        config_json=cfg["config_json"],
    )


async def _authenticate_widget(
    token: str,
    settings: Settings,
) -> _ResolvedConnection | None:
    """Authenticate via widget session token (anonymous SDK visitors).

    Skips user DB lookup — resolves tenant directly from the token.
    Returns a _ResolvedConnection on success, None on failure.
    """
    from app.infrastructure.database.connection import AsyncSessionLocal

    try:
        payload = verify_widget_token(
            token=token,
            secret_key=settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
        )
    except AuthError:
        return None

    async with AsyncSessionLocal() as session:
        tenant_repo = SQLTenantRepository(session)
        tenant = await tenant_repo.get_by_id(payload.tenant_id)

        if not tenant or tenant.status != TenantStatus.ACTIVE:
            return None

        cfg = _extract_tenant_config(tenant)

    return _ResolvedConnection(
        tenant_id=payload.tenant_id,
        user_id="",  # Anonymous — conversations stored with NULL user_id
        temperature=cfg["temperature"],
        blocklist=cfg["blocklist"],
        chat_model=cfg["chat_model"],
        embedding_model=cfg["embedding_model"],
        chat_provider=cfg["chat_provider"],
        gemini_api_key=cfg["gemini_api_key"],
        embedding_provider=cfg["embedding_provider"],
        gemini_embedding_api_key=cfg["gemini_embedding_api_key"],
        agent_config=cfg["agent_config"],
        config_json=cfg["config_json"],
    )


@router.websocket("/api/v1/ws/chat")
async def websocket_chat(
    websocket: WebSocket,
    token: str = Query(default=""),
) -> None:
    """WebSocket endpoint for streaming chat responses.

    Authentication flow:
        1. Client connects with ``?token=<jwt_or_ws_token>`` query param
        2. If token starts with ``ws_``: widget auth (no user lookup)
        3. Otherwise: standard JWT auth (user + tenant lookup)
        4. Server accepts connection and registers with ConnectionManager
        5. Client sends message JSON, server streams response frames

    Args:
        websocket: The incoming WebSocket connection.
        token: JWT access token or ws_-prefixed widget token.
    """
    # ── Authentication ───────────────────────────────────────────
    if not token:
        await websocket.close(code=4001, reason="Missing authentication token")
        return

    settings = get_settings()

    # Dual auth: detect ws_ prefix for widget tokens
    if token.startswith(WIDGET_TOKEN_PREFIX):
        conn = await _authenticate_widget(token, settings)
        if conn is None:
            await websocket.close(code=4001, reason="Invalid or expired widget token")
            return
    else:
        conn = await _authenticate_jwt(token, settings)
        if conn is None:
            await websocket.close(code=4001, reason="Invalid or expired token")
            return

    tenant_id = conn.tenant_id
    user_id = conn.user_id

    # ── Connection ───────────────────────────────────────────────
    ws_manager = getattr(websocket.app.state, "ws_manager", None)
    chat_service = getattr(websocket.app.state, "chat_service", None)

    if ws_manager is None or chat_service is None:
        await websocket.close(code=4500, reason="Service unavailable")
        return

    await ws_manager.connect(websocket, tenant_id=tenant_id, user_id=user_id)

    try:
        while True:
            # Wait for client message
            data = await websocket.receive_json()
            message = data.get("message", "")
            conversation_id = data.get("conversation_id")

            # ── Stop command: client can abort an active stream ──
            if data.get("type") == "stop":
                logger.info(
                    "ws_client_stop_requested",
                    tenant_id=tenant_id,
                    user_id=user_id,
                )
                # Nothing to actively cancel here; the stop is
                # handled inside the streaming loop below via a
                # concurrent receive check.
                continue

            if not message or not message.strip():
                await ws_manager.send_json(
                    websocket,
                    {
                        "type": "error",
                        "data": {"message": "Message cannot be empty"},
                    },
                )
                continue

            # Stream response with cancellation support.
            # We run two concurrent tasks:
            #   1. The stream consumer (sends frames to the client)
            #   2. A "wait for stop" listener (watches for stop command or disconnect)
            # Whichever finishes first cancels the other.

            stop_event = asyncio.Event()

            async def _stream_to_client(
                _message: str = message,
                _conversation_id: str | None = conversation_id,
                _stop: asyncio.Event = stop_event,
            ) -> None:
                """Consume the LLM stream and forward frames to the client."""
                stream_gen = chat_service.stream_message(
                    message=_message,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    conversation_id=_conversation_id,
                    temperature=conn.temperature,
                    tenant_blocklist=conn.blocklist,
                    tenant_chat_model=conn.chat_model,
                    tenant_embedding_model=conn.embedding_model,
                    tenant_chat_provider=conn.chat_provider,
                    tenant_gemini_api_key=conn.gemini_api_key,
                    tenant_embedding_provider=conn.embedding_provider,
                    tenant_gemini_embedding_api_key=conn.gemini_embedding_api_key,
                    tenant_agent_config=conn.agent_config,
                    tenant_config_json=conn.config_json,
                )
                try:
                    async for frame in stream_gen:
                        if _stop.is_set():
                            break
                        await ws_manager.send_json(websocket, frame)
                except (LLMError, SupportForgeError) as e:
                    logger.error(
                        "ws_stream_error",
                        error=str(e),
                        tenant_id=tenant_id,
                        user_id=user_id,
                    )
                    if not _stop.is_set():
                        await ws_manager.send_json(
                            websocket,
                            {
                                "type": "error",
                                "data": {"message": str(e.message)},
                            },
                        )
                except Exception as e:
                    logger.error(
                        "ws_unexpected_error",
                        error=str(e),
                        error_type=type(e).__name__,
                        tenant_id=tenant_id,
                        user_id=user_id,
                    )
                    if not _stop.is_set():
                        await ws_manager.send_json(
                            websocket,
                            {
                                "type": "error",
                                "data": {"message": "An unexpected error occurred"},
                            },
                        )
                finally:
                    # Explicitly close the generator chain so that
                    # OllamaAdapter.stream() exits its try/finally,
                    # closing the httpx response and the TCP connection.
                    # Ollama detects the disconnect and stops GPU work.
                    await stream_gen.aclose()

            async def _wait_for_stop(
                _stop: asyncio.Event = stop_event,
            ) -> None:
                """Listen for a stop command while the stream is active."""
                try:
                    while not _stop.is_set():
                        msg = await websocket.receive_json()
                        if msg.get("type") == "stop":
                            logger.info(
                                "ws_stop_during_stream",
                                tenant_id=tenant_id,
                                user_id=user_id,
                            )
                            _stop.set()
                            return
                        else:
                            await ws_manager.send_json(
                                websocket,
                                {
                                    "type": "error",
                                    "data": {"message": "Please wait for the current response to complete"},
                                },
                            )
                except WebSocketDisconnect:
                    _stop.set()
                    raise

            stream_task = asyncio.create_task(_stream_to_client())
            stop_task = asyncio.create_task(_wait_for_stop())

            try:
                done, pending = await asyncio.wait(
                    {stream_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                    with contextlib.suppress(
                        asyncio.CancelledError, WebSocketDisconnect,
                    ):
                        await task

                # Re-raise WebSocketDisconnect or other exceptions
                for task in done:
                    exc = task.exception() if not task.cancelled() else None
                    if isinstance(exc, WebSocketDisconnect):
                        raise exc
                    elif exc is not None:
                        logger.error(
                            "ws_task_crashed",
                            exc_info=exc,
                            tenant_id=tenant_id,
                            user_id=user_id,
                        )
                        if task is stream_task and not stop_event.is_set():
                            raise exc

                # If stopped, send a done frame so the UI knows
                if stop_event.is_set():
                    await ws_manager.send_json(
                        websocket,
                        {
                            "type": "done",
                            "data": {
                                "conversation_id": conversation_id,
                                "stopped": True,
                            },
                        },
                    )
            except WebSocketDisconnect:
                raise

    except WebSocketDisconnect:
        logger.info("ws_client_disconnected", tenant_id=tenant_id, user_id=user_id)
    finally:
        ws_manager.disconnect(websocket, tenant_id=tenant_id)

