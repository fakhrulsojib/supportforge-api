"""Tool loop — multi-round LLM ↔ tool execution cycle.

Called by BOTH ``process_message`` and ``stream_message`` after the
RAG graph completes.  The loop asks the LLM whether tools are needed,
executes them, feeds results back, and repeats until the LLM stops
requesting tools or ``max_rounds`` is reached.

The LLM's text response when it stops calling tools is **discarded** —
the caller generates the final answer separately (streaming or
non-streaming) with tool results in context.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from app.domain.interfaces.llm_provider import LLMProvider
from app.rag.prompt_builder import build_rag_messages, format_rag_context
from app.rag.tools.executor import ToolExecutor

logger = structlog.get_logger(__name__)


def _safe_json_dumps(obj: Any) -> str:
    """Serialize to JSON, falling back to str() for non-serializable objects.
    Truncates to 100KB to prevent memory exhaustion and context window bloat
    if the LLM hallucinates massive arguments.
    """
    try:
        res = json.dumps(obj)
    except (TypeError, ValueError):
        res = json.dumps(str(obj))
        
    if len(res) > 100000:
        # If we truncate, it won't be valid JSON anymore, but it's safer than crashing.
        res = res[:100000] + '... [TRUNCATED]'
    return res


from collections.abc import AsyncGenerator

async def run_tool_loop(
    state: dict[str, Any],
    tools: list[Any],
    llm_provider: LLMProvider,
    executor: ToolExecutor,
    *,
    system_prompt: str,
    conversation_history: list[dict[str, str]] | None = None,
    chat_model: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Run the tool decision loop.

    Called by BOTH ``process_message`` and ``stream_message`` — single
    codepath for tool decisions.

    Flow:
        1. Ask LLM (via ``generate_with_tools``) whether to call any tools
        2. If tool calls → execute them → feed results back → loop
        3. If no tool calls (LLM returns text) → discard text → return state
        4. If ``escalate`` tool called → set ``should_escalate=True`` → return
        5. If ``max_rounds`` reached → return state

    Args:
        state: RAGState dict from the graph (has ``query``, ``relevant_docs``, etc.)
        tools: List of tool instances (WebhookTool and/or BuiltinEscalateTool).
        llm_provider: LLM provider for ``generate_with_tools()`` calls.
        executor: ToolExecutor with safety guardrails.
        system_prompt: Pre-built system prompt (from ``build_system_prompt()``).
        conversation_history: Previous conversation messages.
        chat_model: Model to use for tool calls.

    Returns:
        AsyncGenerator yielding intermediate tool frames and finally the state dict.
    """
    tool_defs = [t.definition.to_openai_format() for t in tools]
    tool_map = {t.definition.name: t for t in tools}

    # Build initial messages: system + history + RAG context + user query
    context = format_rag_context(state.get("relevant_docs", []))
    messages = build_rag_messages(
        query=state.get("query", ""),
        context=context,
        history_messages=conversation_history,
        system_prompt=system_prompt,
    )

    all_tool_calls: list[dict[str, Any]] = state.get("tool_calls", [])
    all_tool_results: list[dict[str, Any]] = state.get("tool_results", [])

    for round_num in range(executor.max_rounds):
        # Ask LLM: should I call a tool?
        response = await llm_provider.generate_with_tools(
            messages=messages,
            tools=tool_defs,
            model=chat_model,
        )

        if not response.tool_calls:
            # LLM decided no (more) tools needed.
            # If tools were executed in previous rounds, the LLM's response
            # already incorporates tool results — save it as the answer.
            if all_tool_calls and response.content:
                state["tool_answer"] = response.content
            logger.debug(
                "tool_loop_no_tools",
                round=round_num,
                had_tool_calls=bool(all_tool_calls),
                content_preview=(response.content or "")[:100],
            )
            break

        # Circuit breaker: detect identical failed tool calls
        current_calls_fingerprint = [
            (tc.name, _safe_json_dumps(tc.arguments))
            for tc in response.tool_calls
        ]
        
        # Check if the exact same calls were made in the previous round
        if round_num > 0 and len(response.tool_calls) > 0:
            prev_tool_msgs = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
            if prev_tool_msgs:
                prev_num_calls = len(prev_tool_msgs[-1].get("tool_calls", []))
                if prev_num_calls > 0 and prev_num_calls == len(response.tool_calls):
                    prev_calls = all_tool_calls[-prev_num_calls:]
                    prev_fingerprint = [(c["name"], _safe_json_dumps(c["arguments"])) for c in prev_calls]
                    
                    if current_calls_fingerprint == prev_fingerprint:
                        logger.warning(
                            "tool_loop_stuck",
                            round=round_num,
                            reason="Identical tool calls repeated across rounds.",
                        )
                        state["should_escalate"] = True
                        state["escalation_reason"] = "Assistant is stuck repeating identical tool calls."
                        break

        # Execute each tool call
        assistant_tool_calls = []
        tool_result_messages = []

        for tc in response.tool_calls:
            tool = tool_map.get(tc.name)
            if not tool:
                logger.warning("tool_not_found", name=tc.name)
                assistant_tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": _safe_json_dumps(tc.arguments),
                    },
                })
                tool_result_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"error": f"Unknown tool: {tc.name}"}),
                })
                continue

            # Check for built-in escalate tool
            if tc.name == "escalate":
                reason = tc.arguments.get("reason", "")
                state["should_escalate"] = True
                state["escalation_reason"] = reason
                logger.info("tool_escalation", reason=reason)
                all_tool_calls.append({
                    "id": tc.id, "name": tc.name, "arguments": tc.arguments,
                })
                assistant_tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": _safe_json_dumps(tc.arguments),
                    },
                })
                assistant_msg = response.raw_message if getattr(response, "raw_message", None) else {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": assistant_tool_calls,
                }
                messages.append(assistant_msg)
                state["tool_calls"] = all_tool_calls
                state["tool_results"] = all_tool_results
                state["tool_messages"] = [
                    m for m in messages
                    if m.get("role") == "tool" or m.get("tool_calls")
                ]
                yield {"type": "state", "data": state}
                return  # Exit loop immediately on escalation

            yield {"type": "tool_start", "data": {"name": tc.name}}

            # Execute the tool
            result = await executor.execute(tool, tc.arguments)

            yield {"type": "tool_result", "data": {"name": tc.name, "success": result.success}}

            # Record the interaction
            all_tool_calls.append({
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
            })
            all_tool_results.append({
                "tool_call_id": tc.id,
                "name": tc.name,
                "result": result.data,
                "success": result.success,
                "error": result.error,
                "execution_time_ms": result.execution_time_ms,
            })

            logger.info(
                "tool_executed_in_loop",
                tool=tc.name,
                round=round_num,
                success=result.success,
                time_ms=result.execution_time_ms,
            )

            # Collect for consolidated assistant message
            assistant_tool_calls.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": _safe_json_dumps(tc.arguments),
                },
            })

            result_content = (
                result.data if result.success
                else {"error": result.error or "Tool execution failed"}
            )
            tool_result_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": _safe_json_dumps(result_content),
            })

        # Append ONE assistant message with all tool calls, then all results
        if assistant_tool_calls:
            assistant_msg = response.raw_message if getattr(response, "raw_message", None) else {
                "role": "assistant",
                "content": response.content,
                "tool_calls": assistant_tool_calls,
            }
            messages.append(assistant_msg)
            messages.extend(tool_result_messages)

        state["tool_round"] = round_num + 1

    # Store tool interaction data
    state["tool_calls"] = all_tool_calls
    state["tool_results"] = all_tool_results

    # Store tool messages for conversation history persistence
    # (assistant + tool role messages from the loop)
    state["tool_messages"] = [
        m for m in messages
        if m.get("role") == "tool" or m.get("tool_calls")
    ]

    yield {"type": "state", "data": state}

