# Tests

This is not a comprehensive test suite. It's a small set of tests that prove specific invariants — things the loop *must* guarantee for correct behavior, independent of which model is on the other end.

## What we're testing and why

The interesting failure modes in an agent loop fall into two layers:

**Layer 1 — structural invariants.** These don't require a live LLM. They verify that the plumbing holds: context trimming never corrupts the message sequence, parallel tool calls are always fully resolved before the next round, and a crashing tool handler can't take down the loop. If these break, no amount of prompt engineering fixes it.

**Layer 2 — behavioral evals.** These require a real model. Does the agent actually use its tools to find answers rather than hallucinating? Does it pick the right tool for the job? 11 questions across 5 categories (chat, loop, tools, config, grep) — each with a ground-truth answer verifiable from the repo itself.

## Unit tests

`test_chat_flow.py` — basic Chat class mechanics: appending turns, single and multi-tool sequences, the message structure the OpenAI API expects.

`test_chat_scenarios.py` — message structure across complete multi-round workflows. Simulates the turn sequences `agent_loop` produces (user → tool rounds → final assistant) and asserts counts and roles are correct. No LLM involved.

`test_loop_invariants.py` — three Layer 1 proofs:
- **Atomic group preservation**: trimming drops an `assistant[tool_calls]` + its tool results together, never one without the other. Verified by building a chat with two tool-call groups, trimming to a budget that forces the first group out, then walking the resulting message list for orphaned tool messages.
- **Parallel result ordering**: `_execute_tool_calls` injects all results before returning, in invocation order, even when handlers finish out-of-order. Verified by racing three async handlers with different sleep durations and asserting position, not just presence.
- **Exception isolation**: a handler that raises produces an `Error:` tool message in chat; it doesn't propagate out of the loop. Verified on `_run_single_tool` directly, then again through `_execute_tool_calls` with a mixed success/failure call list.

## Integration tests (Layer 2 evals)

`test_evals.py` — 11 behavioral questions sent to a live model. Each question has a known correct answer in the repo. Two assertions per question: the model called at least one tool (no hallucination without evidence), and the response contains the expected string(s).

Results are written to `tests/benchmarks/` after each run as `2026_05_08_1431_qwen3-14b.json` — date-first so they sort chronologically. Each run produces a separate file so results accumulate across models for comparison. The JSON includes per-question responses, pass/fail, tool call flag, latency, and a summary broken down by category.

### Configs

Model configs for benchmarking live in `tests/configs/`. The project root `config.yaml` is for general use; the files below are specifically for eval runs.

| Config | Model |
|--------|-------|
| `tests/configs/config.qwen3-0.6b.yaml` | qwen3:0.6b |
| `tests/configs/config.qwen3-4b.yaml` | qwen3:4b |
| `tests/configs/config.qwen3-8b.yaml` | qwen3:8b |
| `tests/configs/config.qwen3-14b.yaml` | qwen3:14b |

### Running

```bash
# Unit tests only (no model needed)
pytest tests/unit/ -v

# Full benchmark across all 4 models
pytest tests/integration/test_evals.py -v -s --config=tests/configs/config.qwen3-0.6b.yaml
pytest tests/integration/test_evals.py -v -s --config=tests/configs/config.qwen3-4b.yaml
pytest tests/integration/test_evals.py -v -s --config=tests/configs/config.qwen3-8b.yaml
pytest tests/integration/test_evals.py -v -s --config=tests/configs/config.qwen3-14b.yaml
```

Evals skip automatically if the model endpoint is unreachable — safe to run in CI without Ollama.

## Results

Pass/fail is determined by substring matching against known correct answers — a structural check. The `response` field in each JSON contains the full model answer for qualitative review. LLM-as-judge and human-in-the-loop validation are the next step for evaluating answer quality beyond keyword presence.

_Hardware: Ollama on NVIDIA RTX 4070 Super, Fedora Linux 43._

| Model | Pass rate | Tool call rate | chat | config | grep | loop | project | tools |
|-------|-----------|----------------|------|--------|------|------|---------|-------|
| qwen3:0.6b | 9.1% | 27.3% | 0/2 | 0/2 | 0/2 | 0/2 | 0/1 | 1/2 |
| qwen3:4b | 72.7% | 90.9% | 1/2 | 2/2 | 2/2 | 1/2 | 0/1 | 2/2 |
| qwen3:8b | 90.9% | 90.9% | 1/2 | 2/2 | 2/2 | 2/2 | 1/1 | 2/2 |
| qwen3:14b | 100% | 100% | 2/2 | 2/2 | 2/2 | 2/2 | 1/1 | 2/2 |

Raw benchmark files: [`tests/benchmarks/`](benchmarks/)
