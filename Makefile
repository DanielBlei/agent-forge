OLLAMA_CONFIG  := config.ollama.yaml
OLLAMA_MODEL   ?= qwen3:14b

VLLM_CONFIG    := config.example.yaml
VLLM_MODEL     := mistralai/Ministral-3-3B-Reasoning-2512
VLLM_VENV      := .venv-vllm

PROTO_URL      := https://raw.githubusercontent.com/DanielBlei/go-to-rag/main/proto/rag/v1/rag.proto
PROTO_OUT      := agent_forge/_gen

.PHONY: help run serve run-ollama proto test clean

help:
	@echo "agent-forge"
	@echo ""
	@echo "  make run          Run agent (vLLM, requires make serve first)"
	@echo "  make serve        Start vLLM server"
	@echo "  make run-ollama   Start Ollama + run agent"
	@echo "  make proto        Regenerate gRPC stubs from go-to-rag proto"
	@echo "  make test         Run all tests"
	@echo "  make clean        Remove build artefacts"

run:
	python main.py --config $(VLLM_CONFIG)

serve:
	$(VLLM_VENV)/bin/vllm serve $(VLLM_MODEL) \
		--tokenizer_mode mistral \
		--config_format mistral \
		--load_format mistral \
		--enable-auto-tool-choice \
		--tool-call-parser mistral \
		--port 8000 \
		--gpu-memory-utilization 0.90 \
		--max-model-len 16384 \
		--max-num-seqs 1

run-ollama:
	@ollama serve > /dev/null 2>&1 & until curl -sf http://localhost:11434 > /dev/null; do sleep 0.5; done
	@ollama pull $(OLLAMA_MODEL)
	python main.py --config $(OLLAMA_CONFIG) --debug

proto:
	@mkdir -p /tmp/_proto_src/rag/v1
	curl -sSL $(PROTO_URL) -o /tmp/_proto_src/rag/v1/rag.proto
	python -m grpc_tools.protoc \
		-I /tmp/_proto_src \
		--python_out=$(PROTO_OUT) \
		--grpc_python_out=$(PROTO_OUT) \
		rag/v1/rag.proto
	python -c "\
		import pathlib; \
		p = pathlib.Path('$(PROTO_OUT)/rag/v1/rag_pb2_grpc.py'); \
		p.write_text(p.read_text().replace('from rag.v1 import rag_pb2', 'from agent_forge._gen.rag.v1 import rag_pb2'))"
	@rm -rf /tmp/_proto_src

test:
	python -m pytest tests/ -q

clean:
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name ".ruff_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	@rm -f test_grep_*.py
