"""
Integration tests for agent-forge.

These tests verify the complete workflow and integration between components.
"""

import json

import pytest

from agent_forge.chat import Chat


def test_chat_multi_tool_integration():
    """Test that demonstrates the expected multi-tool workflow in chat."""
    chat = Chat(system="You are a helpful assistant.")
    chat.user("Perform a complex task requiring multiple tools")

    # Simulate the complete agent_loop workflow:

    # 1. First API call - model decides to use tool1
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

    # Execute tool1 and add result (this would be done by _execute_tool_calls)
    tool1_result = {"result": "tool1_done"}
    chat.tool("call1", json.dumps(tool1_result))

    # 2. Second API call - model decides to use tool2
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

    # Execute tool2 and add result
    tool2_result = {"result": "tool2_done"}
    chat.tool("call2", json.dumps(tool2_result))

    # 3. Third API call - model decides to use tool3
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

    # Execute tool3 and add result
    tool3_result = {"result": "tool3_done"}
    chat.tool("call3", json.dumps(tool3_result))

    # 4. Final API call - model provides text response
    chat.assistant("All tools completed successfully")

    # Verify the complete workflow
    assert len(chat) == 8  # user + 3*(assistant+tool) + final assistant

    messages = chat.to_messages()
    assert len(messages) == 9  # system + 8 turns

    # Verify tool calls were properly recorded
    tool_call_messages = [msg for msg in messages if msg.get("role") == "tool"]
    assert len(tool_call_messages) == 3

    # Verify assistant messages with tool calls
    assistant_with_tools = [msg for msg in messages if msg.get("role") == "assistant" and msg.get("tool_calls")]
    assert len(assistant_with_tools) == 3

    # Verify final message has no tool calls
    final_message = messages[-1]
    assert final_message["role"] == "assistant"
    assert final_message.get("tool_calls") is None
    assert "All tools completed successfully" in final_message["content"]


def test_agent_loop_expected_behavior():
    """Test that demonstrates the expected behavior of agent_loop."""
    chat = Chat(system="You are a helpful assistant.")
    chat.user("Complex multi-step task")

    # This test demonstrates what agent_loop should do:

    # Initial state
    assert len(chat) == 1  # Just the user message
    initial_turns = len(chat)

    # Simulate one iteration of agent_loop:
    # 1. Model returns tool calls
    chat.assistant(
        content="Using a tool",
        tool_calls=[
            {"id": "call1", "type": "function", "function": {"name": "tool1", "arguments": json.dumps({"step": 1})}}
        ],
    )

    # 2. Tool is executed and result added
    chat.tool("call1", json.dumps({"result": "done"}))

    # After one iteration, we should have more turns
    assert len(chat) == initial_turns + 2  # +assistant +tool

    # The loop would continue if model returns more tool calls
    # This demonstrates the expected continuation behavior


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
