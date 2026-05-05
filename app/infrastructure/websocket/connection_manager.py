"""WebSocket connection manager — tracks active connections per tenant.

Provides connection lifecycle management (connect, disconnect) and
message broadcasting scoped by tenant for multi-tenant isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from fastapi import WebSocket

logger = structlog.get_logger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections grouped by tenant.

    Thread-safe for single-process async usage (all operations are
    synchronous dict/set mutations within the event loop).

    Attributes:
        _connections: Mapping of tenant_id → set of active WebSockets.
    """

    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = {}

    @property
    def active_connection_count(self) -> int:
        """Total number of active connections across all tenants."""
        return sum(len(conns) for conns in self._connections.values())

    def get_tenant_connections(self, tenant_id: str) -> list[WebSocket]:
        """Return all active connections for a given tenant.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            List of active WebSocket connections (empty if none).
        """
        return list(self._connections.get(tenant_id, set()))

    async def connect(
        self,
        websocket: WebSocket,
        tenant_id: str,
        user_id: str,
    ) -> None:
        """Accept a WebSocket connection and register it.

        Args:
            websocket: The incoming WebSocket.
            tenant_id: Tenant the user belongs to.
            user_id: Authenticated user identifier (for logging).
        """
        await websocket.accept()

        if tenant_id not in self._connections:
            self._connections[tenant_id] = set()
        self._connections[tenant_id].add(websocket)

        logger.info(
            "ws_connected",
            tenant_id=tenant_id,
            user_id=user_id,
            total_connections=self.active_connection_count,
        )

    def disconnect(self, websocket: WebSocket, tenant_id: str) -> None:
        """Remove a WebSocket connection from tracking.

        Safe to call even if the websocket was never connected
        (no-op in that case).

        Args:
            websocket: The disconnecting WebSocket.
            tenant_id: Tenant the connection belonged to.
        """
        conns = self._connections.get(tenant_id)
        if conns is not None:
            conns.discard(websocket)
            if not conns:
                del self._connections[tenant_id]

        logger.info(
            "ws_disconnected",
            tenant_id=tenant_id,
            total_connections=self.active_connection_count,
        )

    async def send_json(self, websocket: WebSocket, data: dict[str, Any]) -> None:
        """Send a JSON message to a specific WebSocket.

        Args:
            websocket: Target WebSocket connection.
            data: JSON-serializable data to send.
        """
        await websocket.send_json(data)

    async def broadcast_to_tenant(
        self,
        tenant_id: str,
        data: dict[str, Any],
    ) -> None:
        """Broadcast a JSON message to all connections in a tenant.

        Failures on individual connections are logged but do not
        prevent delivery to remaining healthy connections.

        Args:
            tenant_id: Target tenant.
            data: JSON-serializable data to broadcast.
        """
        for ws in self.get_tenant_connections(tenant_id):
            try:
                await ws.send_json(data)
            except Exception:
                logger.warning(
                    "ws_broadcast_send_failed",
                    tenant_id=tenant_id,
                    exc_info=True,
                )
