from __future__ import annotations

from typing import Any

import grpc
from grpc import aio as grpc_aio
from openai.types.chat import ChatCompletionToolParam

from agent_forge._gen.rag.v1 import rag_pb2, rag_pb2_grpc
from agent_forge.logger import get_logger

logger = get_logger(__name__)

_THINK_MODES: dict[str, int] = {
    "disabled": rag_pb2.THINK_MODE_DISABLED,
    "hidden": rag_pb2.THINK_MODE_HIDDEN,
}


def init_rag_tools(
    endpoint: str,
) -> tuple[list[ChatCompletionToolParam], dict[str, Any], grpc_aio.Channel]:
    """Create gRPC channel and return (tool_definitions, tool_handlers, channel).

    The caller owns the channel and must close it (await channel.close()) when done.
    """
    channel = grpc_aio.insecure_channel(endpoint)
    stub = rag_pb2_grpc.RAGServiceStub(channel)

    async def ask_rag(
        question: str,
        top_k: int = 5,
        think_mode: str | None = None,
    ) -> dict[str, Any]:
        req = rag_pb2.AskRequest(question=question, top_k=top_k)
        if think_mode is not None:
            req.think_mode = _THINK_MODES.get(think_mode, rag_pb2.THINK_MODE_UNSPECIFIED)

        answer_parts: list[str] = []
        thinking_parts: list[str] = []
        try:
            async for chunk in stub.Ask(req):
                if chunk.answer:
                    answer_parts.append(chunk.answer)
                if chunk.thinking:
                    thinking_parts.append(chunk.thinking)
        except grpc.RpcError as e:
            logger.error("ask_rag failed: %s", e)
            return {"success": False, "error": str(e)}

        result: dict[str, Any] = {"success": True, "answer": "".join(answer_parts)}
        if thinking_parts:
            result["thinking"] = "".join(thinking_parts)
        return result

    async def retrieve_chunks(question: str, top_k: int = 5) -> dict[str, Any]:
        req = rag_pb2.RetrieveChunksRequest(question=question, top_k=top_k)
        try:
            resp = await stub.RetrieveChunks(req)
        except grpc.RpcError as e:
            logger.error("retrieve_chunks failed: %s", e)
            return {"success": False, "error": str(e)}

        return {
            "success": True,
            "chunks": [
                {
                    "text": c.text,
                    "source": c.source,
                    "score": round(c.score, 4),
                    "chunk_index": c.chunk_index,
                }
                for c in resp.chunks
            ],
            "count": len(resp.chunks),
        }

    tool_definitions: list[ChatCompletionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": "ask_rag",
                "description": (
                    "Query the knowledge base with a question. The RAG engine retrieves "
                    "relevant document chunks and generates a grounded answer. Use this "
                    "when you need a synthesised answer from stored knowledge."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask the knowledge base.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of chunks to retrieve (default: 5).",
                        },
                        "think_mode": {
                            "type": "string",
                            "enum": ["disabled", "hidden"],
                            "description": (
                                "Controls RAG model reasoning. 'disabled' skips thinking entirely "
                                "(faster). 'hidden' reasons internally but omits thinking from the "
                                "response. Omit to use the server default."
                            ),
                        },
                    },
                    "required": ["question"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "retrieve_chunks",
                "description": (
                    "Retrieve raw scored document chunks from the knowledge base without LLM "
                    "generation. Use this when you want to inspect source material directly, "
                    "check relevance scores, or synthesise an answer yourself."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Query used to find relevant chunks.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of chunks to return (default: 5).",
                        },
                    },
                    "required": ["question"],
                },
            },
        },
    ]

    tool_handlers: dict[str, Any] = {
        "ask_rag": ask_rag,
        "retrieve_chunks": retrieve_chunks,
    }

    logger.info("RAG tools initialised — endpoint: %s", endpoint)
    return tool_definitions, tool_handlers, channel
