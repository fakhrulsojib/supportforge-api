"""Tests for WebSocket connection manager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.infrastructure.websocket.connection_manager import ConnectionManager


class TestConnectionManager:
    """Test suite for ConnectionManager."""

    def test_initial_state(self) -> None:
        """Manager starts with no active connections."""
        manager = ConnectionManager()
        assert manager.active_connection_count == 0
        assert manager.get_tenant_connections("any-tenant") == []

    @pytest.mark.asyncio
    async def test_connect_accepts_and_tracks(self) -> None:
        """connect() should accept the websocket and track it."""
        manager = ConnectionManager()
        ws = AsyncMock()

        await manager.connect(ws, tenant_id="tenant-1", user_id="user-1")

        ws.accept.assert_called_once()
        assert manager.active_connection_count == 1
        assert len(manager.get_tenant_connections("tenant-1")) == 1

    @pytest.mark.asyncio
    async def test_connect_multiple_tenants(self) -> None:
        """Connections from different tenants are tracked separately."""
        manager = ConnectionManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        await manager.connect(ws1, tenant_id="tenant-1", user_id="user-1")
        await manager.connect(ws2, tenant_id="tenant-2", user_id="user-2")

        assert manager.active_connection_count == 2
        assert len(manager.get_tenant_connections("tenant-1")) == 1
        assert len(manager.get_tenant_connections("tenant-2")) == 1

    @pytest.mark.asyncio
    async def test_connect_multiple_users_same_tenant(self) -> None:
        """Multiple users in same tenant tracked in same set."""
        manager = ConnectionManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        await manager.connect(ws1, tenant_id="tenant-1", user_id="user-1")
        await manager.connect(ws2, tenant_id="tenant-1", user_id="user-2")

        assert manager.active_connection_count == 2
        assert len(manager.get_tenant_connections("tenant-1")) == 2

    @pytest.mark.asyncio
    async def test_disconnect_removes_connection(self) -> None:
        """disconnect() should remove the connection from tracking."""
        manager = ConnectionManager()
        ws = AsyncMock()

        await manager.connect(ws, tenant_id="tenant-1", user_id="user-1")
        assert manager.active_connection_count == 1

        manager.disconnect(ws, tenant_id="tenant-1")
        assert manager.active_connection_count == 0
        assert len(manager.get_tenant_connections("tenant-1")) == 0

    @pytest.mark.asyncio
    async def test_disconnect_unknown_websocket_is_noop(self) -> None:
        """Disconnecting a websocket that was never connected should not error."""
        manager = ConnectionManager()
        ws = AsyncMock()

        # Should not raise
        manager.disconnect(ws, tenant_id="tenant-1")
        assert manager.active_connection_count == 0

    @pytest.mark.asyncio
    async def test_disconnect_cleans_empty_tenant(self) -> None:
        """After all connections for a tenant disconnect, tenant key is cleaned up."""
        manager = ConnectionManager()
        ws = AsyncMock()

        await manager.connect(ws, tenant_id="tenant-1", user_id="user-1")
        manager.disconnect(ws, tenant_id="tenant-1")

        # Tenant key should be removed entirely
        assert manager.get_tenant_connections("tenant-1") == []

    @pytest.mark.asyncio
    async def test_send_json(self) -> None:
        """send_json() delegates to websocket.send_json()."""
        manager = ConnectionManager()
        ws = AsyncMock()

        await manager.connect(ws, tenant_id="tenant-1", user_id="user-1")
        await manager.send_json(ws, {"type": "token", "data": "hello"})

        ws.send_json.assert_called_once_with({"type": "token", "data": "hello"})

    @pytest.mark.asyncio
    async def test_broadcast_to_tenant(self) -> None:
        """broadcast_to_tenant() sends to all connections in that tenant."""
        manager = ConnectionManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        ws3 = AsyncMock()

        await manager.connect(ws1, tenant_id="tenant-1", user_id="user-1")
        await manager.connect(ws2, tenant_id="tenant-1", user_id="user-2")
        await manager.connect(ws3, tenant_id="tenant-2", user_id="user-3")

        message = {"type": "done", "data": {}}
        await manager.broadcast_to_tenant("tenant-1", message)

        ws1.send_json.assert_called_once_with(message)
        ws2.send_json.assert_called_once_with(message)
        ws3.send_json.assert_not_called()
