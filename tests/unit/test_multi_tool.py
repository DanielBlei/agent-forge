"""
Unit tests for multi-tool behavior and agent loop functionality.

This module tests the expected flow of multiple tool calls in sequence,
demonstrating how the agent_loop should handle complex multi-step operations.
"""

import json

import pytest

from agent_forge.chat import Chat


def mock_read_file(path: str) -> dict:
    """Mock read_file tool."""
    return {"success": True, "content": f"Mock content of {path}"}


def mock_glob_files(pattern: str) -> dict:
    """Mock glob_files tool."""
    return {"success": True, "matches": [f"file1_{pattern}", f"file2_{pattern}"]}


def mock_grep_files(pattern: str) -> dict:
    """Mock grep_files tool."""
    return {"success": True, "matches": [{"file": "test.py", "line": 1, "content": "match found"}]}


def test_multi_tool_chat_structure():
    """Test the expected chat structure for multiple tool calls."""
    chat = Chat(system="You are a helpful assistant that uses tools to answer questions.")
    chat.user("Find all Python files in the project, then read the main.py file, and search for 'async' usage")

    # Expected tool definitions that would be used
    tools = [
        {
            "type": "function",
            "function": {
                "name": "glob_files",
                "description": "Find files matching a glob pattern",
                "parameters": {
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep_files",
                "description": "Search files for a pattern",
                "parameters": {
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"],
                },
            },
        },
    ]

    tool_handlers = {"glob_files": mock_glob_files, "read_file": mock_read_file, "grep_files": mock_grep_files}

    # Verify initial chat state
    assert len(chat) == 1  # Just the user message
    assert chat.token_count() > 0

    messages = chat.to_messages()
    assert len(messages) == 2  # system + user
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Python files" in messages[1]["content"]


def test_expected_agent_loop_flow():
    """Test the expected flow of agent_loop with multiple tools."""
    chat = Chat(system="You are a helpful assistant.")
    chat.user("Complex task requiring multiple tools")

    # Simulate the expected agent_loop flow:

    # 1. Model decides to use glob_files first
    chat.assistant(
        content="Let me find the relevant files first",
        tool_calls=[
            {
                "id": "call1",
                "type": "function",
                "function": {"name": "glob_files", "arguments": json.dumps({"pattern": "**.py"})},
            }
        ],
    )

    # 2. Execute glob_files and add result
    result1 = mock_glob_files("**.py")
    chat.tool("call1", json.dumps(result1))

    # 3. Model decides to use read_file next
    chat.assistant(
        content="Now let me read the main file",
        tool_calls=[
            {
                "id": "call2",
                "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": "main.py"})},
            }
        ],
    )

    # 4. Execute read_file and add result
    result2 = mock_read_file("main.py")
    chat.tool("call2", json.dumps(result2))

    # 5. Model decides to use grep_files next
    chat.assistant(
        content="Let me search for specific content",
        tool_calls=[
            {
                "id": "call3",
                "type": "function",
                "function": {"name": "grep_files", "arguments": json.dumps({"pattern": "async"})},
            }
        ],
    )

    # 6. Execute grep_files and add result
    result3 = mock_grep_files("async")
    chat.tool("call3", json.dumps(result3))

    # 7. Model provides final answer
    chat.assistant("Here's the complete analysis based on all the tools used")

    # Verify the complete flow
    assert len(chat) == 8  # user + 3*(assistant+tool) + final assistant
    messages = chat.to_messages()
    assert len(messages) == 9  # system + 8 turns

    # Verify tool calls were properly recorded
    tool_call_messages = [msg for msg in messages if msg.get("role") == "tool"]
    assert len(tool_call_messages) == 3

    # Verify assistant messages with tool calls
    assistant_with_tools = [msg for msg in messages if msg.get("role") == "assistant" and msg.get("tool_calls")]
    assert len(assistant_with_tools) == 3


def test_agent_loop_termination():
    """Test that agent_loop should terminate when model returns text without tool calls."""
    chat = Chat(system="You are a helpful assistant.")
    chat.user("Simple question")

    # Model provides direct answer without tools
    chat.assistant("Here's the answer to your question")

    assert len(chat) == 2  # user + assistant
    messages = chat.to_messages()
    assert len(messages) == 3  # system + user + assistant

    # No tool calls should be present
    assert not any(msg.get("tool_calls") for msg in messages)
    assert not any(msg.get("role") == "tool" for msg in messages)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
