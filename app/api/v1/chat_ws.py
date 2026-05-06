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

import structlog
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.config import get_settings
from app.core.exceptions import AuthError, LLMError, SupportForgeError
from app.core.security import verify_token
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

    tenant_temperature = 0.7  # default
    async with AsyncSessionLocal() as session:
        user_repo = SQLUserRepository(session)
        user = await user_repo.get_by_id(payload.user_id)
        if not user:
            await websocket.close(code=4001, reason="User not found")
            return

        # Read per-tenant LLM temperature from config_json
        tenant_repo = SQLTenantRepository(session)
        tenant = await tenant_repo.get_by_id(user.tenant_id)
        if tenant and tenant.config_json:
            raw = tenant.config_json.get("temperature")
            if isinstance(raw, (int, float)) and 0.0 <= float(raw) <= 1.0:
                tenant_temperature = float(raw)

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

            if not message or not message.strip():
                await ws_manager.send_json(
                    websocket,
                    {
                        "type": "error",
                        "data": {"message": "Message cannot be empty"},
                    },
                )
                continue

            # Stream response
            try:
                async for frame in chat_service.stream_message(
                    message=message,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    temperature=tenant_temperature,
                ):
                    await ws_manager.send_json(websocket, frame)
            except (LLMError, SupportForgeError) as e:
                logger.error(
                    "ws_stream_error",
                    error=str(e),
                    tenant_id=tenant_id,
                    user_id=user_id,
                )
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
                await ws_manager.send_json(
                    websocket,
                    {
                        "type": "error",
                        "data": {"message": "An unexpected error occurred"},
                    },
                )

    except WebSocketDisconnect:
        logger.info("ws_client_disconnected", tenant_id=tenant_id, user_id=user_id)
    finally:
        ws_manager.disconnect(websocket, tenant_id=tenant_id)
