"""
Layer 2 behavioral evals: does the agent use its tools to answer correctly?

Each case sends a question to a live Agent and checks:
  1. At least one tool was called (no hallucination without evidence).
  2. The response contains every expected substring.

Results are written to results/evals_{model}_{timestamp}.json by conftest.
Run with a different config to benchmark a different model:

    AGENT_FORGE_CONFIG=config.ollama.yaml pytest tests/integration/test_evals.py -v -s
"""

from __future__ import annotations

import time

import pytest

# ---------------------------------------------------------------------------
# Eval cases
# ---------------------------------------------------------------------------

EVALS = [
    {
        "id": 1,
        "category": "chat",
        "description": "MSG_OVERHEAD constant value",
        "question": "What is the value of the _MSG_OVERHEAD constant in chat.py?",
        "expected": ["4"],
    },
    {
        "id": 2,
        "category": "loop",
        "description": "Retry attempts for transient errors",
        "question": "How many retry attempts does agent_loop make when a transient API error occurs?",
        "expected": ["3"],
    },
    {
        "id": 3,
        "category": "chat",
        "description": "Default token encoding",
        "question": "What token encoding does the Chat class use by default?",
        "expected": ["cl100k_base"],
    },
    {
        "id": 4,
        "category": "project",
        "description": "Required Python version",
        "question": "What is the minimum Python version required by this project?",
        "expected": ["3.12"],
    },
    {
        "id": 5,
        "category": "loop",
        "description": "Retried exception types",
        "question": "Which exception types trigger automatic retry in agent_loop?",
        "expected": ["APIConnectionError", "APITimeoutError"],
    },
    {
        "id": 6,
        "category": "grep",
        "description": "File that defines TOOL_DEFINITIONS",
        "question": "Which file defines the TOOL_DEFINITIONS list?",
        "expected": ["tools.py"],
    },
    {
        "id": 8,
        "category": "tools",
        "description": "Excluded directories in file search",
        "question": "Which directory names are excluded from file search in tools.py?",
        "expected": [".git", ".venv", "__pycache__"],
    },
    {
        "id": 9,
        "category": "config",
        "description": "Model in config.ollama.yaml",
        "question": "What model name is configured in config.ollama.yaml?",
        "expected": ["qwen3"],
    },
    {
        "id": 10,
        "category": "config",
        "description": "context_limit in config.example.yaml",
        "question": "What is the context_limit value in config.example.yaml?",
        "expected": ["16384"],
    },
    {
        "id": 11,
        "category": "grep",
        "description": "File where _trim_context is defined",
        "question": "In which file is the _trim_context function defined?",
        "expected": ["loop.py"],
    },
    {
        "id": 12,
        "category": "tools",
        "description": "Number of default tool handlers",
        "question": "How many tool handlers are registered in TOOL_HANDLERS in tools.py?",
        "expected": ["4"],
    },
]


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("case", EVALS, ids=[f"{e['id']:02d}_{e['description'].replace(' ', '_')}" for e in EVALS])
async def test_eval(case: dict, agent, benchmark_results: list) -> None:
    print(f"\n[{case['id']:02d}/{len(EVALS)}] [{case['category']}] {case['description']}")
    print(f"  Q: {case['question']}")

    t0 = time.perf_counter()
    response = await agent.run(case["question"])
    latency = round(time.perf_counter() - t0, 3)

    tool_messages = [m for m in agent.chat.to_messages() if m.get("role") == "tool"]
    tool_called = len(tool_messages) > 0

    expected = case["expected"] if isinstance(case["expected"], list) else [case["expected"]]
    passed = all(e.lower() in response.lower() for e in expected)

    benchmark_results.append(
        {
            "id": case["id"],
            "category": case["category"],
            "description": case["description"],
            "question": case["question"],
            "expected": case["expected"],
            "passed": passed,
            "tool_called": tool_called,
            "latency_s": latency,
            "response": response,
        }
    )

    assert tool_called, f"no tool called for: {case['question']!r}"
    assert passed, f"expected {expected!r} not found in response: {response[:300]!r}"