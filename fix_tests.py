import re

with open('tests/unit/test_chat_ws.py', 'r') as f:
    content = f.read()

# Make all def receive_json_side_effect() into async def
content = re.sub(r'def receive_json_side_effect\(\):', r'async def receive_json_side_effect():', content)

# Change return asyncio.sleep(0, result=...) to return ...
content = re.sub(r'return asyncio\.sleep\(0, result=(.*?)\)', r'return \1', content)

# Change return asyncio.sleep(0.01, result=...) to await asyncio.sleep(0.01)\n            return ...
content = re.sub(r'return asyncio\.sleep\((0\.01), result=(.*?)\)', r'await asyncio.sleep(\1)\n            return \2', content)

# Change return asyncio.sleep(0, side_effect=...) to raise ...
content = re.sub(r'return asyncio\.sleep\(0, side_effect=(.*?)\)', r'raise \1', content)

# Change return wait_future to return await wait_future
content = re.sub(r'return wait_future', r'return await wait_future', content)

# Change MagicMock(side_effect=receive_json_side_effect) to AsyncMock(...)
content = re.sub(r'MagicMock\(side_effect=receive_json_side_effect\)', r'AsyncMock(side_effect=receive_json_side_effect)', content)

with open('tests/unit/test_chat_ws.py', 'w') as f:
    f.write(content)

