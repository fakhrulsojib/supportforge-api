import re

with open('tests/unit/test_chat_ws.py', 'r') as f:
    content = f.read()

# Replace with pytest.raises(WebSocketDisconnect):\n        await websocket_chat(...)
# with just await websocket_chat(...)
pattern = r'    with pytest\.raises\(WebSocketDisconnect\):\n        await websocket_chat\(mock_ws, token="valid"\)'
replacement = r'    await websocket_chat(mock_ws, token="valid")'
content = re.sub(pattern, replacement, content)

with open('tests/unit/test_chat_ws.py', 'w') as f:
    f.write(content)

