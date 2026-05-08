"""
Unit tests for Chat message structure across complete multi-round scenarios.

Verifies that message counts, roles, and tool_call linkage are correct after
simulating the kind of turn sequences agent_loop produces — without calling
the real loop or any LLM.
"""

import json

import pytest

from agent_forge.chat import Chat


def test_multi_round_tool_sequence():
    """Three sequential tool-call rounds produce the correct message structure."""
    chat = Chat(system="You are a helpful assistant.")
    chat.user("Perform a complex task requiring multiple tools")

    chat.assistant(
        content="Let me use tool1 first",
        tool_calls=[
            {
                "id": "call1",
                "type": "function",
                "function": {"name": "tool1", "arguments": json.dumps({"arg": "step1"})},
            }
        ],
    )
    chat.tool("call1", json.dumps({"result": "tool1_done"}))

    chat.assistant(
        content="Now let me use tool2",
        tool_calls=[
            {
                "id": "call2",
                "type": "function",
                "function": {"name": "tool2", "arguments": json.dumps({"arg": "step2"})},
            }
        ],
    )
    chat.tool("call2", json.dumps({"result": "tool2_done"}))

    chat.assistant(
        content="Finally, let me use tool3",
        tool_calls=[
            {
                "id": "call3",
                "type": "function",
                "function": {"name": "tool3", "arguments": json.dumps({"arg": "step3"})},
            }
        ],
    )
    chat.tool("call3", json.dumps({"result": "tool3_done"}))

    chat.assistant("All tools completed successfully")

    assert len(chat) == 8  # user + 3*(assistant+tool) + final assistant

    messages = chat.to_messages()
    assert len(messages) == 9  # system + 8 turns

    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 3

    assistant_with_tools = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(assistant_with_tools) == 3

    final = messages[-1]
    assert final["role"] == "assistant"
    assert final.get("tool_calls") is None
    assert "All tools completed successfully" in final["content"]


def test_single_round_appends_turns():
    """One tool-call round increments the turn count by exactly two."""
    chat = Chat(system="You are a helpful assistant.")
    chat.user("Complex multi-step task")
    assert len(chat) == 1

    chat.assistant(
        content="Using a tool",
        tool_calls=[
            {"id": "call1", "type": "function", "function": {"name": "tool1", "arguments": json.dumps({"step": 1})}}
        ],
    )
    chat.tool("call1", json.dumps({"result": "done"}))

    assert len(chat) == 3  # user + assistant + tool


if __name__ == "__main__":
    pytest.main([__file__, "-v"])