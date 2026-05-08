"""
Unit tests proving behavioral invariants of the agent loop.

Tests are scoped to Layer 1: structural guarantees of context management and
tool execution, independent of any live LLM or external service.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_forge.chat import Chat
from agent_forge.loop import _execute_tool_calls, _run_single_tool


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _assert_no_orphans(messages: list) -> None:
    """Assert the atomic-group invariant on a message list.

    Two checks:
    1. Every tool message's tool_call_id must appear in a preceding
       assistant message's tool_calls list.
    2. Every assistant message that carries tool_calls must be immediately
       followed by exactly the matching set of tool-result messages.
    """
    declared: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                declared.add(tc["id"])

    for msg in messages:
        if msg.get("role") == "tool":
            assert msg["tool_call_id"] in declared, (
                f"Orphaned tool message with tool_call_id={msg['tool_call_id']!r} — "
                "no preceding assistant[tool_calls] message declares this id"
            )

    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            expected: set[str] = {tc["id"] for tc in msg["tool_calls"]}
            j = i + 1
            found: set[str] = set()
            while j < len(messages) and messages[j].get("role") == "tool":
                found.add(messages[j]["tool_call_id"])
                j += 1
            assert found == expected, (
                f"assistant[tool_calls] at index {i} not immediately followed by matching tool messages — "
                f"expected={expected}, found={found}"
            )


# ---------------------------------------------------------------------------
# Test 1
# ---------------------------------------------------------------------------


def test_atomic_group_preserved_under_trim() -> None:
    """Trimming never separates an assistant[tool_calls] from its tool results."""
    # Constant token counter: every message costs 10 + 4 overhead = 14 tokens exactly,
    # so we can reason about budgets without touching the real tiktoken encoder.
    chat = Chat(system="S", token_counter=lambda _: 10)

    # Group 1 — 14 tokens, will be dropped
    chat.user("U1")

    # Group 2 — 3 messages × 14 = 42 tokens, will be dropped
    chat.assistant(
        tool_calls=[
            {"id": "tcA", "type": "function", "function": {"name": "fa", "arguments": "{}"}},
            {"id": "tcB", "type": "function", "function": {"name": "fb", "arguments": "{}"}},
        ]
    )
    chat.tool("tcA", "result_A")
    chat.tool("tcB", "result_B")

    # Group 3 — 14 tokens, kept
    chat.user("U2")

    # Group 4 — 2 messages × 14 = 28 tokens, kept
    chat.assistant(
        tool_calls=[
            {"id": "tcC", "type": "function", "function": {"name": "fc", "arguments": "{}"}},
        ]
    )
    chat.tool("tcC", "result_C")

    # system(14) + G1(14) + G2(42) + G3(14) + G4(28) = 112
    assert chat.token_count() == 112

    # Budget 56: drops G1 (112→98>56), then G2 (98→56), stops.
    chat.trim_to_budget(56)
    assert chat.token_count() == 56

    messages = chat.to_messages()
    assert len(messages) == 4  # system + U2 + assistant(tcC) + tool(tcC)

    _assert_no_orphans(messages)


# ---------------------------------------------------------------------------
# Test 2
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_tool_results_all_injected_before_next_round() -> None:
    """_execute_tool_calls injects all results in invocation order even when handlers finish out-of-order."""

    async def slow_handler() -> str:
        await asyncio.sleep(0.05)
        return "slow_result"

    async def fast_handler() -> str:
        await asyncio.sleep(0.01)
        return "fast_result"

    async def medium_handler() -> str:
        await asyncio.sleep(0.03)
        return "medium_result"

    tool_calls = [
        {"id": "call_slow", "type": "function", "function": {"name": "slow_tool", "arguments": "{}"}},
        {"id": "call_fast", "type": "function", "function": {"name": "fast_tool", "arguments": "{}"}},
        {"id": "call_medium", "type": "function", "function": {"name": "medium_tool", "arguments": "{}"}},
    ]
    tool_handlers = {
        "slow_tool": slow_handler,
        "fast_tool": fast_handler,
        "medium_tool": medium_handler,
    }

    chat = Chat(system="S")
    await _execute_tool_calls(tool_calls, tool_handlers, chat)

    tool_messages = [m for m in chat.to_messages() if m.get("role") == "tool"]
    assert len(tool_messages) == 3

    ids_present = {m["tool_call_id"] for m in tool_messages}
    assert ids_present == {"call_slow", "call_fast", "call_medium"}

    # asyncio.gather preserves invocation order regardless of completion order
    assert tool_messages[0]["tool_call_id"] == "call_slow"
    assert tool_messages[1]["tool_call_id"] == "call_fast"
    assert tool_messages[2]["tool_call_id"] == "call_medium"


# ---------------------------------------------------------------------------
# Test 3
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_handler_exception_produces_error_message_and_loop_continues() -> None:
    """A raising handler surfaces as an error tool message; the loop does not crash."""

    def boom_handler() -> None:
        raise RuntimeError("boom")

    # Part A: _run_single_tool returns (id, error_string) without propagating the exception
    tc_err = {"id": "call_err", "type": "function", "function": {"name": "boom_tool", "arguments": "{}"}}
    tool_id, result = await _run_single_tool(tc_err, {"boom_tool": boom_handler})

    assert tool_id == "call_err"
    assert result.startswith("Error:")
    assert "boom" in result

    # Part B: _execute_tool_calls writes both messages — one normal, one error
    def ok_handler() -> dict:
        return {"ok": True}

    tool_calls = [
        {"id": "call_ok", "type": "function", "function": {"name": "ok_tool", "arguments": "{}"}},
        {"id": "call_err2", "type": "function", "function": {"name": "boom_tool", "arguments": "{}"}},
    ]
    tool_handlers = {"ok_tool": ok_handler, "boom_tool": boom_handler}

    chat = Chat(system="S")
    await _execute_tool_calls(tool_calls, tool_handlers, chat)

    tool_messages = [m for m in chat.to_messages() if m.get("role") == "tool"]
    assert len(tool_messages) == 2

    ok_msg = next(m for m in tool_messages if m["tool_call_id"] == "call_ok")
    assert json.loads(ok_msg["content"]) == {"ok": True}

    err_msg = next(m for m in tool_messages if m["tool_call_id"] == "call_err2")
    assert err_msg["content"].startswith("Error:")
    assert "boom" in err_msg["content"]