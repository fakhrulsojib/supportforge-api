import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import WebSocket, WebSocketDisconnect
from app.api.v1.chat_ws import websocket_chat, _ResolvedConnection
from app.core.exceptions import LLMError, SupportForgeError

@pytest.fixture
def resolved_conn():
    return _ResolvedConnection(
        tenant_id="tenant-test",
        user_id="user-test",
        temperature=0.7,
        blocklist=["bad"],
        chat_model="test-chat",
        embedding_model="test-embed",
        chat_provider="test-provider",
    )

@pytest.fixture
def mock_ws():
    ws = AsyncMock(spec=WebSocket)
    ws.app = MagicMock()
    
    ws_manager = AsyncMock()
    ws_manager.connect = AsyncMock()
    ws_manager.disconnect = MagicMock()
    ws_manager.send_json = AsyncMock()
    ws.app.state.ws_manager = ws_manager
    
    chat_service = MagicMock()
    
    # We will mock the stream_message return value per test.
    ws.app.state.chat_service = chat_service
    
    return ws

@pytest.mark.asyncio
async def test_missing_token(mock_ws):
    await websocket_chat(mock_ws, token="")
    mock_ws.close.assert_awaited_once_with(code=4001, reason="Missing authentication token")

@pytest.mark.asyncio
@patch("app.api.v1.chat_ws.get_settings")
@patch("app.api.v1.chat_ws._authenticate_widget")
async def test_invalid_widget_token(mock_auth, mock_settings, mock_ws):
    mock_auth.return_value = None
    await websocket_chat(mock_ws, token="ws_invalid")
    mock_ws.close.assert_awaited_once_with(code=4001, reason="Invalid or expired widget token")

@pytest.mark.asyncio
@patch("app.api.v1.chat_ws.get_settings")
@patch("app.api.v1.chat_ws._authenticate_jwt")
async def test_invalid_jwt_token(mock_auth, mock_settings, mock_ws):
    mock_auth.return_value = None
    await websocket_chat(mock_ws, token="invalid_jwt")
    mock_ws.close.assert_awaited_once_with(code=4001, reason="Invalid or expired token")

@pytest.mark.asyncio
@patch("app.api.v1.chat_ws.get_settings")
@patch("app.api.v1.chat_ws._authenticate_jwt")
async def test_missing_services(mock_auth, mock_settings, mock_ws, resolved_conn):
    mock_auth.return_value = resolved_conn
    mock_ws.app.state.ws_manager = None
    await websocket_chat(mock_ws, token="valid")
    mock_ws.close.assert_awaited_once_with(code=4500, reason="Service unavailable")

@pytest.mark.asyncio
@patch("app.api.v1.chat_ws.get_settings")
@patch("app.api.v1.chat_ws._authenticate_jwt")
async def test_client_disconnect_immediately(mock_auth, mock_settings, mock_ws, resolved_conn):
    mock_auth.return_value = resolved_conn
    # simulate disconnect on first message receive
    mock_ws.receive_json.side_effect = WebSocketDisconnect()
    
    await websocket_chat(mock_ws, token="valid")
    
    mock_ws.app.state.ws_manager.connect.assert_awaited_once_with(mock_ws, tenant_id="tenant-test", user_id="user-test")
    mock_ws.app.state.ws_manager.disconnect.assert_called_once_with(mock_ws, tenant_id="tenant-test")

@pytest.mark.asyncio
@patch("app.api.v1.chat_ws.get_settings")
@patch("app.api.v1.chat_ws._authenticate_jwt")
async def test_empty_message_sends_error(mock_auth, mock_settings, mock_ws, resolved_conn):
    mock_auth.return_value = resolved_conn
    
    # send one empty message, then disconnect to break loop
    mock_ws.receive_json.side_effect = [{"message": "  "}, WebSocketDisconnect()]
    
    await websocket_chat(mock_ws, token="valid")
    
    mock_ws.app.state.ws_manager.send_json.assert_awaited_once_with(
        mock_ws, {"type": "error", "data": {"message": "Message cannot be empty"}}
    )

@pytest.mark.asyncio
@patch("app.api.v1.chat_ws.get_settings")
@patch("app.api.v1.chat_ws._authenticate_jwt")
async def test_stop_message_ignores_if_not_streaming(mock_auth, mock_settings, mock_ws, resolved_conn):
    mock_auth.return_value = resolved_conn
    
    # send a stop message, then disconnect to break loop
    mock_ws.receive_json.side_effect = [{"type": "stop"}, WebSocketDisconnect()]
    
    await websocket_chat(mock_ws, token="valid")
    
    # should just log and continue, not sending any json
    mock_ws.app.state.ws_manager.send_json.assert_not_called()

@pytest.mark.asyncio
@patch("app.api.v1.chat_ws.get_settings")
@patch("app.api.v1.chat_ws._authenticate_jwt")
async def test_successful_stream(mock_auth, mock_settings, mock_ws, resolved_conn):
    mock_auth.return_value = resolved_conn
    
    # Setup for receive_json inside wait_for_stop
    # first receive_json is main loop receiving user message
    # second receive_json is wait_for_stop waiting for stop msg. We'll make it block forever.
    # actually, if wait_for_stop blocks, stream_task completes first.
    # to avoid wait_for_stop throwing an error, we make it block.
    wait_future = asyncio.Future()
    
    async def receive_json_side_effect():
        if receive_json_side_effect.call_count == 0:
            receive_json_side_effect.call_count += 1
            return {"message": "hello", "conversation_id": "c1"}
        elif receive_json_side_effect.call_count == 1:
            receive_json_side_effect.call_count += 1
            return await wait_future # Blocks until stream finishes
        else:
            raise WebSocketDisconnect()

    receive_json_side_effect.call_count = 0
    mock_ws.receive_json = AsyncMock(side_effect=receive_json_side_effect)
    
    # Mock generator
    async def mock_stream_gen(**kwargs):
        yield {"type": "token", "data": "abc"}
        # when generator ends, we resolve the wait_future to throw WebSocketDisconnect
        # to break the main loop gracefully.
        asyncio.get_running_loop().call_soon(lambda: wait_future.set_exception(WebSocketDisconnect()))
    
    mock_ws.app.state.chat_service.stream_message.return_value = mock_stream_gen()
    
    await websocket_chat(mock_ws, token="valid")
    
    # Assert generator was called with right arguments
    mock_ws.app.state.chat_service.stream_message.assert_called_once()
    kwargs = mock_ws.app.state.chat_service.stream_message.call_args.kwargs
    assert kwargs["message"] == "hello"
    assert kwargs["tenant_id"] == "tenant-test"
    assert kwargs["conversation_id"] == "c1"
    
    mock_ws.app.state.ws_manager.send_json.assert_awaited_once_with(
        mock_ws, {"type": "token", "data": "abc"}
    )

@pytest.mark.asyncio
@patch("app.api.v1.chat_ws.get_settings")
@patch("app.api.v1.chat_ws._authenticate_jwt")
async def test_stream_llmerror(mock_auth, mock_settings, mock_ws, resolved_conn):
    mock_auth.return_value = resolved_conn
    
    # Receive "hello" then wait forever
    wait_future = asyncio.Future()
    async def receive_json_side_effect():
        if receive_json_side_effect.call_count == 0:
            receive_json_side_effect.call_count += 1
            return {"message": "hello"}
        elif receive_json_side_effect.call_count == 1:
            receive_json_side_effect.call_count += 1
            return await wait_future
        else:
            raise WebSocketDisconnect()
    
    receive_json_side_effect.call_count = 0
    mock_ws.receive_json = AsyncMock(side_effect=receive_json_side_effect)
    
    async def mock_stream_gen(**kwargs):
        yield {"type": "token", "data": "abc"}
        raise LLMError("llm failed")
        
    mock_ws.app.state.chat_service.stream_message.return_value = mock_stream_gen()
    
    # Wait future should throw disconnect at end to terminate loop
    async def resolve_wait():
        await asyncio.sleep(0.01)
        wait_future.set_exception(WebSocketDisconnect())
    
    asyncio.create_task(resolve_wait())
    
    await websocket_chat(mock_ws, token="valid")
        
    mock_ws.app.state.ws_manager.send_json.assert_any_call(
        mock_ws, {"type": "error", "data": {"message": "llm failed"}}
    )

@pytest.mark.asyncio
@patch("app.api.v1.chat_ws.get_settings")
@patch("app.api.v1.chat_ws._authenticate_jwt")
async def test_stream_unexpected_error(mock_auth, mock_settings, mock_ws, resolved_conn):
    mock_auth.return_value = resolved_conn
    
    wait_future = asyncio.Future()
    async def receive_json_side_effect():
        if receive_json_side_effect.call_count == 0:
            receive_json_side_effect.call_count += 1
            return {"message": "hello"}
        elif receive_json_side_effect.call_count == 1:
            receive_json_side_effect.call_count += 1
            return await wait_future
        else:
            raise WebSocketDisconnect()
    
    receive_json_side_effect.call_count = 0
    mock_ws.receive_json = AsyncMock(side_effect=receive_json_side_effect)
    
    async def mock_stream_gen(**kwargs):
        yield {"type": "token", "data": "abc"}
        raise ValueError("boom")
        
    mock_ws.app.state.chat_service.stream_message.return_value = mock_stream_gen()
    
    async def resolve_wait():
        await asyncio.sleep(0.01)
        wait_future.set_exception(WebSocketDisconnect())
    
    asyncio.create_task(resolve_wait())
    
    await websocket_chat(mock_ws, token="valid")
        
    mock_ws.app.state.ws_manager.send_json.assert_any_call(
        mock_ws, {"type": "error", "data": {"message": "An unexpected error occurred"}}
    )

@pytest.mark.asyncio
@patch("app.api.v1.chat_ws.get_settings")
@patch("app.api.v1.chat_ws._authenticate_jwt")
async def test_stream_task_crashes_with_exception(mock_auth, mock_settings, mock_ws, resolved_conn):
    mock_auth.return_value = resolved_conn
    
    wait_future = asyncio.Future()
    async def receive_json_side_effect():
        if receive_json_side_effect.call_count == 0:
            receive_json_side_effect.call_count += 1
            return {"message": "hello"}
        else:
            return await wait_future
            
    receive_json_side_effect.call_count = 0
    mock_ws.receive_json = AsyncMock(side_effect=receive_json_side_effect)
    
    async def mock_stream_gen(**kwargs):
        yield {"type": "token", "data": "abc"}
        
    mock_ws.app.state.chat_service.stream_message.return_value = mock_stream_gen()
    
    # send_json raises ValueError
    mock_ws.app.state.ws_manager.send_json.side_effect = ValueError("send failed")
    
    # We expect websocket_chat to raise ValueError
    with pytest.raises(ValueError, match="send failed"):
        await websocket_chat(mock_ws, token="valid")


@pytest.mark.asyncio
@patch("app.api.v1.chat_ws.get_settings")
@patch("app.api.v1.chat_ws._authenticate_jwt")
async def test_wait_for_stop_receives_non_stop(mock_auth, mock_settings, mock_ws, resolved_conn):
    mock_auth.return_value = resolved_conn
    
    stream_future = asyncio.Future()
    async def mock_stream_gen(**kwargs):
        await stream_future
        yield {"type": "token", "data": "abc"}
        
    mock_ws.app.state.chat_service.stream_message.return_value = mock_stream_gen()
    
    async def receive_json_side_effect():
        if receive_json_side_effect.call_count == 0:
            receive_json_side_effect.call_count += 1
            return {"message": "hello"}
        elif receive_json_side_effect.call_count == 1:
            receive_json_side_effect.call_count += 1
            await asyncio.sleep(0.01)
            return {"message": "not a stop"}
        else:
            raise WebSocketDisconnect()
            
    receive_json_side_effect.call_count = 0
    mock_ws.receive_json = AsyncMock(side_effect=receive_json_side_effect)
    
    async def resolve_stream():
        await asyncio.sleep(0.05)
        stream_future.set_result(None)
    
    asyncio.create_task(resolve_stream())
    
    await websocket_chat(mock_ws, token="valid")
        
    mock_ws.app.state.ws_manager.send_json.assert_any_call(
        mock_ws, {"type": "error", "data": {"message": "Please wait for the current response to complete"}}
    )

@pytest.mark.asyncio
@patch("app.api.v1.chat_ws.get_settings")
@patch("app.api.v1.chat_ws._authenticate_jwt")
async def test_wait_for_stop_receives_stop(mock_auth, mock_settings, mock_ws, resolved_conn):
    mock_auth.return_value = resolved_conn
    
    stream_future = asyncio.Future()
    async def mock_stream_gen(**kwargs):
        yield {"type": "token", "data": "first"}
        await stream_future
        yield {"type": "token", "data": "never"}
        
    mock_ws.app.state.chat_service.stream_message.return_value = mock_stream_gen()
    
    async def receive_json_side_effect():
        if receive_json_side_effect.call_count == 0:
            receive_json_side_effect.call_count += 1
            return {"message": "hello", "conversation_id": "c2"}
        elif receive_json_side_effect.call_count == 1:
            receive_json_side_effect.call_count += 1
            await asyncio.sleep(0.01)
            return {"type": "stop"}
        else:
            raise WebSocketDisconnect()
            
    receive_json_side_effect.call_count = 0
    mock_ws.receive_json = AsyncMock(side_effect=receive_json_side_effect)
    
    await websocket_chat(mock_ws, token="valid")
        
    mock_ws.app.state.ws_manager.send_json.assert_any_call(
        mock_ws, {"type": "done", "data": {"conversation_id": "c2", "stopped": True}}
    )

@pytest.mark.asyncio
@patch("app.api.v1.chat_ws.get_settings")
@patch("app.api.v1.chat_ws._authenticate_jwt")
async def test_stop_task_crashes_with_exception(mock_auth, mock_settings, mock_ws, resolved_conn):
    mock_auth.return_value = resolved_conn
    
    stream_future = asyncio.Future()
    async def mock_stream_gen(**kwargs):
        yield {"type": "token", "data": "first"}
        await stream_future
        
    mock_ws.app.state.chat_service.stream_message.return_value = mock_stream_gen()
    
    async def receive_json_side_effect():
        if receive_json_side_effect.call_count == 0:
            receive_json_side_effect.call_count += 1
            return {"message": "hello"}
        elif receive_json_side_effect.call_count == 1:
            receive_json_side_effect.call_count += 1
            # Wait for stop will crash with ValueError when calling receive_json
            raise ValueError("stop task crash")
        else:
            raise WebSocketDisconnect()
            
    receive_json_side_effect.call_count = 0
    mock_ws.receive_json = AsyncMock(side_effect=receive_json_side_effect)
    
    async def resolve_stream():
        await asyncio.sleep(0.05)
        stream_future.set_result(None)
    
    asyncio.create_task(resolve_stream())
    
    # It should catch the ValueError, log it, but NOT raise it. 
    # Instead, the main loop continues, and on the next iteration receive_json raises WebSocketDisconnect.
    await websocket_chat(mock_ws, token="valid")

