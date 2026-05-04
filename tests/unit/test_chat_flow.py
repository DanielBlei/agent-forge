"""
Unit tests for chat flow with multiple tool calls.

This module tests the Chat class's ability to handle multiple tool calls
and results in sequence, which is essential for complex multi-step operations.
"""

import json

import pytest

from agent_forge.chat import Chat


def mock_tool_1(arg: str) -> dict:
    """Mock tool function 1."""
    return {"result": f"tool1_result_{arg}"}


def mock_tool_2(arg: str) -> dict:
    """Mock tool function 2."""
    return {"result": f"tool2_result_{arg}"}


def mock_tool_3(arg: str) -> dict:
    """Mock tool function 3."""
    return {"result": f"tool3_result_{arg}"}


def test_chat_multiple_tool_calls():
    """Test that Chat can handle multiple tool calls and results in sequence."""
    chat = Chat(system="You are a helpful assistant.")
    chat.user("Please use tools to solve this problem")

    # Simulate the flow: model calls tool1
    chat.assistant(
        content="Let me use tool1 first",
        tool_calls=[
            {
                "id": "call1",
                "type": "function",
                "function": {"name": "mock_tool_1", "arguments": json.dumps({"arg": "step1"})},
            }
        ],
    )

    # Execute tool1 and add result
    result1 = mock_tool_1("step1")
    chat.tool("call1", json.dumps(result1))

    # Model calls tool2
    chat.assistant(
        content="Now let me use tool2",
        tool_calls=[
            {
                "id": "call2",
                "type": "function",
                "function": {"name": "mock_tool_2", "arguments": json.dumps({"arg": "step2"})},
            }
        ],
    )

    # Execute tool2 and add result
    result2 = mock_tool_2("step2")
    chat.tool("call2", json.dumps(result2))

    # Model calls tool3
    chat.assistant(
        content="Finally, let me use tool3",
        tool_calls=[
            {
                "id": "call3",
                "type": "function",
                "function": {"name": "mock_tool_3", "arguments": json.dumps({"arg": "step3"})},
            }
        ],
    )

    # Execute tool3 and add result
    result3 = mock_tool_3("step3")
    chat.tool("call3", json.dumps(result3))

    # Model provides final answer
    chat.assistant("Here's the final result based on all tools used")

    # Verify the chat state
    assert len(chat) == 8  # system + user + 3*(assistant+tool) + final assistant
    assert chat.token_count() > 0

    messages = chat.to_messages()
    assert len(messages) == 9  # system + 8 turns

    # Verify message structure
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"] is not None
    assert messages[3]["role"] == "tool"
    assert messages[4]["role"] == "assistant"
    assert messages[4]["tool_calls"] is not None
    assert messages[5]["role"] == "tool"
    assert messages[6]["role"] == "assistant"
    assert messages[6]["tool_calls"] is not None
    assert messages[7]["role"] == "tool"
    assert messages[8]["role"] == "assistant"
    assert messages[8].get("tool_calls") is None


def test_chat_single_tool_call():
    """Test basic single tool call functionality."""
    chat = Chat(system="You are a helpful assistant.")
    chat.user("Use a tool to solve this")

    # Model calls tool
    chat.assistant(
        content="Let me use a tool",
        tool_calls=[
            {
                "id": "call1",
                "type": "function",
                "function": {"name": "mock_tool_1", "arguments": json.dumps({"arg": "test"})},
            }
        ],
    )

    # Execute tool and add result
    result1 = mock_tool_1("test")
    chat.tool("call1", json.dumps(result1))

    # Model provides final answer
    chat.assistant("Done")

    assert len(chat) == 4  # user + assistant + tool + assistant
    messages = chat.to_messages()
    assert len(messages) == 5  # system + 4 turns


def test_chat_no_tools():
    """Test basic chat without tools."""
    chat = Chat(system="You are a helpful assistant.")
    chat.user("Hello")
    chat.assistant("Hi there!")

    assert len(chat) == 2
    messages = chat.to_messages()
    assert len(messages) == 3  # system + user + assistant


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
