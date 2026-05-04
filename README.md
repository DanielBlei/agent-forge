# Agent Forge

> **Status: prototype / experimental** — exploring agentic patterns, not for production use.

Multi-turn LLM agent with streaming tool use, atomic context management, and gRPC RAG integration.

Designed to pair with/explore [go-to-rag](https://github.com/DanielBlei/go-to-rag), a Go RAG engine that exposes
retrieval and generation over gRPC. agent-forge calls it as a first-class tool during conversation.

## Purpose

This repository explores how agentic loops actually work at the implementation level:

- How context windows fill and need to be trimmed without breaking tool-call sequences
- How a local model (Ollama or vLLM) can call out to a separate retrieval service mid-conversation
- How streaming, retries, and context errors compose in a real async loop

The loop, context management, and tool dispatch are all explicit and readable. Deliberately built without
LangChain or LangGraph — the intent is to explore the pillars of agentic workflow at the implementation
level before adopting a framework that abstracts them away.

## Quick start

### Ollama

```bash
ollama pull qwen3:0.6b
make run-ollama
```

### vLLM

```bash
make serve   # starts vLLM on :8000
make run
```

### With go-to-rag

Start the go-to-rag gRPC server — see [go-to-rag serve docs](https://github.com/DanielBlei/go-to-rag/blob/main/docs/serve.md) for setup. Default port is `50051`.

Add to your config:

```yaml
rag:
  endpoint: "localhost:50051"
```

The agent gains two tools: `ask_rag` (retrieval + LLM generation) and `retrieve_chunks` (raw scored chunks).

## Configuration

```yaml
model:
  name: "qwen3:0.6b"
  endpoint: "http://localhost:11434/v1"
  context_limit: 8192

system:
  prompt: "You are a helpful assistant."
  disable_think: false

rag:              # optional — omit if not using go-to-rag
  endpoint: "localhost:50051"
```

**Priority:** CLI flags → environment variables → YAML

| Env var | Effect |
|---|---|
| `AGENT_FORGE_ENDPOINT` | Override model endpoint |
| `OLLAMA_MODEL` | Override Ollama model name |

See `config.example.yaml` for the full schema and `config.ollama.yaml` for an Ollama-ready config.

## Design notes

One round of the agent loop:

```mermaid
flowchart TD
    A([User input]) --> B[Trim context if over budget]
    B --> C[Stream model response]
    C --> D{Tool calls?}
    D -->|Yes| E[Execute tools in parallel]
    E --> F[Inject tool results into chat]
    F --> B
    D -->|No| G([Return final response])
    G -->|Next user message| A
```

**Atomic context trimming** — when the context window fills, turns are dropped oldest-first as atomic
groups. An assistant message carrying tool calls is always dropped together with its tool results —
splitting them leaves orphaned references that cause model errors or hallucinations.

**Token counting** — uses `tiktoken cl100k_base` as a fast approximation. Accurate for OpenAI models;
undercounts ~10–15% on Qwen/Llama with heavy code content.

## Proto regeneration

If the go-to-rag proto changes:

```bash
make proto   # fetches from GitHub, regenerates agent_forge/_gen/
```

## Related

- [go-to-rag](https://github.com/DanielBlei/go-to-rag) — Go RAG engine (gRPC + MCP server)

## License

Apache 2.0
