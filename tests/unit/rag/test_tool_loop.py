"""Tests for the tool loop — multi-round LLM ↔ tool execution cycle."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.interfaces.llm_provider import ToolAwareResponse, ToolCall
from app.rag.tools.base import ToolDefinition, ToolResult
from app.rag.tools.executor import ToolExecutor
from app.rag.tools.resolver import BuiltinEscalateTool
from app.rag.tools.tool_loop import run_tool_loop


def _make_provider(responses: list[ToolAwareResponse]) -> AsyncMock:
    """Create a mock LLM provider that returns the given responses in order."""
    provider = AsyncMock()
    provider.generate_with_tools = AsyncMock(side_effect=responses)
    return provider


def _make_tool(name: str = "check_status") -> MagicMock:
    """Create a mock WebhookTool."""
    tool = MagicMock()
    tool.definition = ToolDefinition(
        name=name, description=f"Mock {name}", parameters={}
    )
    tool.config = MagicMock()
    tool.config.name = name
    tool.config.timeout = 10.0
    tool.execute = AsyncMock(
        return_value=ToolResult(success=True, data={"result": "ok"})
    )
    return tool


def _base_state(**overrides: Any) -> dict[str, Any]:
    """Create a minimal RAGState for testing."""
    state: dict[str, Any] = {
        "query": "Test question",
        "tenant_id": "t-1",
        "relevant_docs": [{"content": "Doc content", "metadata": {"filename": "a.md"}}],
    }
    state.update(overrides)
    return state


class TestRunToolLoop:
    """Tests for the run_tool_loop function."""

    @pytest.mark.asyncio
    async def test_no_tools_called_returns_unchanged_state(self) -> None:
        """LLM decides no tools are needed — state unchanged."""
        provider = _make_provider([
            ToolAwareResponse(content="No tools needed", tool_calls=[]),
        ])
        state = _base_state()
        executor = ToolExecutor(max_rounds=3)
        escalate = BuiltinEscalateTool()

        result = await run_tool_loop(
            state, [escalate], provider, executor,
            system_prompt="Be helpful",
        )
        assert result.get("tool_calls", []) == []
        assert result.get("tool_results", []) == []

    @pytest.mark.asyncio
    async def test_single_tool_call(self) -> None:
        """LLM calls one tool, then stops."""
        mock_tool = _make_tool("check_status")
        provider = _make_provider([
            ToolAwareResponse(
                tool_calls=[ToolCall(id="tc-1", name="check_status", arguments={"id": "123"})],
            ),
            ToolAwareResponse(content="Here's the status", tool_calls=[]),
        ])
        state = _base_state()
        executor = ToolExecutor(max_rounds=3)

        result = await run_tool_loop(
            state, [mock_tool, BuiltinEscalateTool()], provider, executor,
            system_prompt="Be helpful",
        )
        assert len(result.get("tool_calls", [])) == 1
        assert result["tool_calls"][0]["name"] == "check_status"
        assert len(result.get("tool_results", [])) == 1
        assert result["tool_results"][0]["success"] is True
        assert result["tool_round"] == 1

    @pytest.mark.asyncio
    async def test_multi_round_tool_calls(self) -> None:
        """LLM calls tools in 2 rounds before stopping."""
        tool1 = _make_tool("check_availability")
        tool2 = _make_tool("book_appointment")
        tool2.execute = AsyncMock(
            return_value=ToolResult(success=True, data={"booking_id": "b-1"})
        )

        provider = _make_provider([
            ToolAwareResponse(
                tool_calls=[ToolCall(id="tc-1", name="check_availability", arguments={"date": "2026-01-15"})],
            ),
            ToolAwareResponse(
                tool_calls=[ToolCall(id="tc-2", name="book_appointment", arguments={"slot": "10:00"})],
            ),
            ToolAwareResponse(content="Booked!", tool_calls=[]),
        ])
        state = _base_state()
        executor = ToolExecutor(max_rounds=5)

        result = await run_tool_loop(
            state, [tool1, tool2, BuiltinEscalateTool()], provider, executor,
            system_prompt="Be helpful",
        )
        assert len(result.get("tool_calls", [])) == 2
        assert result["tool_round"] == 2

    @pytest.mark.asyncio
    async def test_escalation_via_tool(self) -> None:
        """Escalate tool immediately exits the loop."""
        provider = _make_provider([
            ToolAwareResponse(
                tool_calls=[ToolCall(id="tc-1", name="escalate", arguments={"reason": "Customer angry"})],
            ),
        ])
        state = _base_state()
        executor = ToolExecutor(max_rounds=3)
        escalate = BuiltinEscalateTool()

        result = await run_tool_loop(
            state, [escalate], provider, executor,
            system_prompt="Be helpful",
        )
        assert result["should_escalate"] is True
        assert result["escalation_reason"] == "Customer angry"
        assert len(result.get("tool_calls", [])) == 1

    @pytest.mark.asyncio
    async def test_max_rounds_enforced(self) -> None:
        """Tool loop stops after max_rounds even if LLM keeps calling tools."""
        mock_tool = _make_tool("check")
        # LLM keeps calling tools forever with unique arguments
        provider = _make_provider([
            ToolAwareResponse(
                tool_calls=[ToolCall(id=f"tc-{i}", name="check", arguments={"i": i})],
            )
            for i in range(10)
        ])
        state = _base_state()
        executor = ToolExecutor(max_rounds=2)

        result = await run_tool_loop(
            state, [mock_tool, BuiltinEscalateTool()], provider, executor,
            system_prompt="Be helpful",
        )
        assert result["tool_round"] == 2
        # Only 2 tool calls despite 10 available responses
        assert len(result.get("tool_calls", [])) == 2

    @pytest.mark.asyncio
    async def test_unknown_tool_handled(self) -> None:
        """Unknown tool name is handled gracefully."""
        provider = _make_provider([
            ToolAwareResponse(
                tool_calls=[ToolCall(id="tc-1", name="nonexistent_tool", arguments={})],
            ),
            ToolAwareResponse(content="Fallback", tool_calls=[]),
        ])
        state = _base_state()
        executor = ToolExecutor(max_rounds=3)

        # Only escalate tool is available
        result = await run_tool_loop(
            state, [BuiltinEscalateTool()], provider, executor,
            system_prompt="Be helpful",
        )
        # Should not crash, tool call not recorded in state
        assert len(result.get("tool_calls", [])) == 0

    @pytest.mark.asyncio
    async def test_tool_failure_fed_back_to_llm(self) -> None:
        """When a tool fails, error is fed back so LLM can respond gracefully."""
        mock_tool = _make_tool("failing_tool")
        mock_tool.execute = AsyncMock(
            return_value=ToolResult(success=False, error="API unavailable")
        )
        provider = _make_provider([
            ToolAwareResponse(
                tool_calls=[ToolCall(id="tc-1", name="failing_tool", arguments={})],
            ),
            ToolAwareResponse(content="Sorry, tool failed", tool_calls=[]),
        ])
        state = _base_state()
        executor = ToolExecutor(max_rounds=3)

        result = await run_tool_loop(
            state, [mock_tool, BuiltinEscalateTool()], provider, executor,
            system_prompt="Be helpful",
        )
        assert len(result.get("tool_results", [])) == 1
        assert result["tool_results"][0]["success"] is False
        assert result["tool_results"][0]["error"] == "API unavailable"

    @pytest.mark.asyncio
    async def test_tool_messages_stored_for_persistence(self) -> None:
        """tool_messages contains assistant+tool messages for DB persistence."""
        mock_tool = _make_tool("check")
        provider = _make_provider([
            ToolAwareResponse(
                tool_calls=[ToolCall(id="tc-1", name="check", arguments={"x": 1})],
            ),
            ToolAwareResponse(content="Done", tool_calls=[]),
        ])
        state = _base_state()
        executor = ToolExecutor(max_rounds=3)

        result = await run_tool_loop(
            state, [mock_tool, BuiltinEscalateTool()], provider, executor,
            system_prompt="Be helpful",
        )
        tool_msgs = result.get("tool_messages", [])
        assert len(tool_msgs) >= 2  # assistant + tool
        roles = [m["role"] for m in tool_msgs]
        assert "tool" in roles

    @pytest.mark.asyncio
    async def test_conversation_history_included(self) -> None:
        """Conversation history is passed to the LLM."""
        provider = _make_provider([
            ToolAwareResponse(content="No tools", tool_calls=[]),
        ])
        state = _base_state()
        history = [{"role": "user", "content": "Previous msg"}]
        executor = ToolExecutor(max_rounds=1)

        await run_tool_loop(
            state, [BuiltinEscalateTool()], provider, executor,
            system_prompt="Be helpful",
            conversation_history=history,
        )
        # Verify history was included in the messages
        call_args = provider.generate_with_tools.call_args
        messages = call_args.kwargs.get("messages") or call_args[0][0]
        contents = [m.get("content", "") for m in messages if m.get("role") == "user"]
        assert any("Previous msg" in (c or "") for c in contents)

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_single_round(self) -> None:
        """LLM returns 2+ ToolCall objects in one response."""
        tool_a = _make_tool("tool_a")
        tool_b = _make_tool("tool_b")
        tool_b.execute = AsyncMock(
            return_value=ToolResult(success=True, data={"b": "done"})
        )

        provider = _make_provider([
            ToolAwareResponse(
                tool_calls=[
                    ToolCall(id="tc-1", name="tool_a", arguments={"x": 1}),
                    ToolCall(id="tc-2", name="tool_b", arguments={"y": 2}),
                ],
            ),
            ToolAwareResponse(content="All done", tool_calls=[]),
        ])
        state = _base_state()
        executor = ToolExecutor(max_rounds=3)

        result = await run_tool_loop(
            state, [tool_a, tool_b, BuiltinEscalateTool()], provider, executor,
            system_prompt="Be helpful",
        )
        assert len(result.get("tool_calls", [])) == 2
        assert result["tool_calls"][0]["name"] == "tool_a"
        assert result["tool_calls"][1]["name"] == "tool_b"
        assert result["tool_round"] == 1  # Only one round needed

    @pytest.mark.asyncio
    async def test_chat_model_passed_to_provider(self) -> None:
        """Verify the model parameter flows through to generate_with_tools."""
        provider = _make_provider([
            ToolAwareResponse(content="No tools", tool_calls=[]),
        ])
        state = _base_state()
        executor = ToolExecutor(max_rounds=1)

        await run_tool_loop(
            state, [BuiltinEscalateTool()], provider, executor,
            system_prompt="Be helpful",
            chat_model="gpt-4o-mini",
        )
        call_kwargs = provider.generate_with_tools.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_tool_messages_format_matches_openai_spec(self) -> None:
        """Verify tool_messages have correct OpenAI structure (tool_call_id, role=tool, etc.)."""
        mock_tool = _make_tool("check")
        provider = _make_provider([
            ToolAwareResponse(
                tool_calls=[ToolCall(id="tc-99", name="check", arguments={"a": 1})],
            ),
            ToolAwareResponse(content="Done", tool_calls=[]),
        ])
        state = _base_state()
        executor = ToolExecutor(max_rounds=3)

        result = await run_tool_loop(
            state, [mock_tool, BuiltinEscalateTool()], provider, executor,
            system_prompt="Be helpful",
        )
        tool_msgs = result.get("tool_messages", [])

        # Find the assistant message with tool_calls
        assistant_msgs = [m for m in tool_msgs if m.get("tool_calls")]
        assert len(assistant_msgs) >= 1
        tc_entry = assistant_msgs[0]["tool_calls"][0]
        assert tc_entry["id"] == "tc-99"
        assert tc_entry["type"] == "function"
        assert tc_entry["function"]["name"] == "check"
        assert json.loads(tc_entry["function"]["arguments"]) == {"a": 1}

        # Find the tool response message
        tool_role_msgs = [m for m in tool_msgs if m.get("role") == "tool"]
        assert len(tool_role_msgs) >= 1
        assert tool_role_msgs[0]["tool_call_id"] == "tc-99"
        assert tool_role_msgs[0]["role"] == "tool"
        content = json.loads(tool_role_msgs[0]["content"])
        assert content == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_circuit_breaker_breaks_on_repeated_failed_calls(self) -> None:
        """If LLM repeats identical failed tool calls, the circuit breaker should
        short-circuit and trigger escalation instead of wasting tokens."""
        bad_call = ToolCall(id="tc-bad", name="check_status", arguments={"id": "999"})

        # Round 0: LLM calls tool → fails
        # Round 1: LLM calls EXACT SAME tool with EXACT SAME args → circuit breaker fires
        provider = _make_provider([
            ToolAwareResponse(tool_calls=[bad_call]),
            ToolAwareResponse(tool_calls=[bad_call]),
        ])

        tool = _make_tool("check_status")
        tool.execute = AsyncMock(
            return_value=ToolResult(success=False, data={}, error="Not found")
        )
        escalate = BuiltinEscalateTool()

        state = _base_state()
        result = await run_tool_loop(
            state=state,
            tools=[tool, escalate],
            llm_provider=provider,
            executor=ToolExecutor(max_rounds=5),
            system_prompt="You are a test assistant.",
        )

        assert result.get("should_escalate") is True
        assert "stuck" in result.get("escalation_reason", "").lower()
        # Circuit breaker fires BEFORE executing in round 1, so only 1 execution
        assert tool.execute.call_count == 1

