"""WebSocket chat streaming endpoint — WS /api/v1/ws/chat.

Provides real-time token-by-token streaming of RAG responses.
Authentication is via JWT token in the query string since WebSocket
handshakes do not support custom headers.

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

import structlog
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.config import get_settings
from app.core.exceptions import AuthError, LLMError, SupportForgeError
from app.core.security import verify_token
from app.domain.models.enums import TenantStatus
from app.infrastructure.database.repositories.tenant_repo import SQLTenantRepository
from app.infrastructure.database.repositories.user_repo import SQLUserRepository

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["chat-ws"])


@router.websocket("/api/v1/ws/chat")
async def websocket_chat(
    websocket: WebSocket,
    token: str = Query(default=""),
) -> None:
    """WebSocket endpoint for streaming chat responses.

    Authentication flow:
        1. Client connects with ``?token=<jwt>`` query param
        2. Server validates JWT, extracts user_id + tenant_id
        3. Server accepts connection and registers with ConnectionManager
        4. Client sends message JSON, server streams response frames

    Args:
        websocket: The incoming WebSocket connection.
        token: JWT access token from query string.
    """
    # ── Authentication ───────────────────────────────────────────
    if not token:
        await websocket.close(code=4001, reason="Missing authentication token")
        return

    settings = get_settings()

    try:
        payload = verify_token(
            token=token,
            secret_key=settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
            expected_type="access",
        )
    except AuthError as e:
        await websocket.close(code=4001, reason=str(e.message))
        return

    # Look up user and tenant config in a single session
    from app.infrastructure.database.connection import AsyncSessionLocal

    tenant_temperature = 0.2  # default
    tenant_blocklist: list[str] = []  # default — no blocked terms
    tenant_chat_model: str | None = None  # None → use server default
    tenant_embedding_model: str | None = None  # None → use server default
    tenant_chat_provider: str | None = None  # None → use Ollama
    tenant_gemini_api_key: str | None = None  # None → no Gemini key
    async with AsyncSessionLocal() as session:
        user_repo = SQLUserRepository(session)
        user = await user_repo.get_by_id(payload.user_id)
        if not user:
            await websocket.close(code=4001, reason="User not found")
            return

        # Read per-tenant LLM temperature and moderation blocklist from config_json
        tenant_repo = SQLTenantRepository(session)
        tenant = await tenant_repo.get_by_id(user.tenant_id)

        # ── Tenant status gate ───────────────────────────────────
        # Only ACTIVE tenants can access chat via WebSocket.
        if not tenant or tenant.status != TenantStatus.ACTIVE:
            await websocket.close(
                code=4003,
                reason="Tenant not active — chat access is disabled",
            )
            return

        if tenant and tenant.config_json:
            raw = tenant.config_json.get("temperature")
            if isinstance(raw, (int, float)) and 0.0 <= float(raw) <= 1.0:
                tenant_temperature = float(raw)
            raw_blocklist = tenant.config_json.get("moderation_blocklist")
            if isinstance(raw_blocklist, list):
                tenant_blocklist = [str(t) for t in raw_blocklist if t]
            # Per-tenant model selections (admin-configurable)
            from app.core.tenant_config import resolve_tenant_models
            settings = get_settings()
            tenant_models = resolve_tenant_models(
                tenant.config_json,
                encryption_key=settings.secret_key,
            )
            tenant_chat_model = tenant_models.chat_model
            tenant_embedding_model = tenant_models.embedding_model
            tenant_chat_provider = tenant_models.chat_provider
            tenant_gemini_api_key = tenant_models.gemini_api_key

    tenant_id = user.tenant_id
    user_id = user.id

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

            async def _stream_to_client() -> None:
                """Consume the LLM stream and forward frames to the client."""
                stream_gen = chat_service.stream_message(
                    message=message,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    temperature=tenant_temperature,
                    tenant_blocklist=tenant_blocklist,
                    tenant_chat_model=tenant_chat_model,
                    tenant_embedding_model=tenant_embedding_model,
                    tenant_chat_provider=tenant_chat_provider,
                    tenant_gemini_api_key=tenant_gemini_api_key,
                )
                try:
                    async for frame in stream_gen:
                        if stop_event.is_set():
                            break
                        await ws_manager.send_json(websocket, frame)
                except (LLMError, SupportForgeError) as e:
                    logger.error(
                        "ws_stream_error",
                        error=str(e),
                        tenant_id=tenant_id,
                        user_id=user_id,
                    )
                    if not stop_event.is_set():
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
                    if not stop_event.is_set():
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

            async def _wait_for_stop() -> None:
                """Listen for a stop command while the stream is active."""
                try:
                    while not stop_event.is_set():
                        msg = await websocket.receive_json()
                        if msg.get("type") == "stop":
                            logger.info(
                                "ws_stop_during_stream",
                                tenant_id=tenant_id,
                                user_id=user_id,
                            )
                            stop_event.set()
                            return
                except WebSocketDisconnect:
                    stop_event.set()
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
                    try:
                        await task
                    except (asyncio.CancelledError, WebSocketDisconnect):
                        pass

                # Re-raise WebSocketDisconnect if the stop listener caught it
                for task in done:
                    exc = task.exception() if not task.cancelled() else None
                    if isinstance(exc, WebSocketDisconnect):
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
