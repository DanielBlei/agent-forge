from __future__ import annotations

import functools
import itertools
import json
from collections import deque
from collections.abc import Callable

import tiktoken
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from agent_forge.logger import get_logger

logger = get_logger(__name__)

# Token overhead per message: role label + framing per OpenAI cookbook.
_MSG_OVERHEAD = 4


@functools.cache
def _cl100k() -> tiktoken.Encoding:
    """Lazy-load cl100k_base once — avoids an import-time network/cache hit."""
    return tiktoken.get_encoding("cl100k_base")


def _default_token_counter(text: str) -> int:
    """Approximate counter via cl100k_base (GPT-4 tokenizer).

    Accurate enough for OpenAI models. For Qwen3 it under-counts on CJK text and code.
    Trim_to_budget will be slightly conservative, which is safe.
    For exact counts inject a counter backed by vLLM's POST `/v1/tokenize`.
    """
    return len(_cl100k().encode(text))


def _msg_text(msg: ChatCompletionMessageParam) -> str:
    """Extract all countable text from a message dict.

    Handles content (str or multimodal list) and the tool_calls JSON blob
    that live inside assistant messages — both consume real context tokens.
    """
    parts: list[str] = []

    content = msg.get("content")
    if isinstance(content, list):
        parts.extend(b.get("text", "") for b in content if isinstance(b, dict))
    elif content:
        parts.append(str(content))

    tool_calls = msg.get("tool_calls")
    if tool_calls:
        parts.append(json.dumps(tool_calls, default=str))

    return " ".join(parts)


class Chat:
    """Ordered conversation context backed by OpenAI message TypedDicts.

    The system message is pinned and never trimmed. Turns are appended via
    typed helpers and dropped oldest-first when over the token budget.

    Args:
        system: The system prompt — pinned, never trimmed.
        token_counter: Callable (text -> int). Defaults to cl100k_base
            approximation. Inject a vLLM-backed counter for exact counts:

                async def vllm_counter(text: str) -> int:
                    r = await client.post("/v1/tokenize", json={"model": model, "prompt": text})
                    return r.json()["count"]
    """

    def __init__(
        self,
        system: str,
        token_counter: Callable[[str], int] = _default_token_counter,
    ) -> None:
        self._system: ChatCompletionSystemMessageParam = {"role": "system", "content": system}
        self._turns: deque[ChatCompletionMessageParam] = deque()
        self._count = token_counter
        self._total_tokens: int = self._msg_tokens(self._system)

    def _msg_tokens(self, msg: ChatCompletionMessageParam) -> int:
        return self._count(_msg_text(msg)) + _MSG_OVERHEAD

    def _append(self, msg: ChatCompletionMessageParam) -> None:
        self._turns.append(msg)
        self._total_tokens += self._msg_tokens(msg)

    # ------------------------------------------------------------------
    # Append helpers
    # ------------------------------------------------------------------

    def user(self, content: str) -> None:
        msg: ChatCompletionUserMessageParam = {"role": "user", "content": content}
        self._append(msg)

    def assistant(self, content: str | None = None, tool_calls: list | None = None) -> None:
        """Append an assistant turn.

        content is None when the model emits a tool call with no accompanying
        text — passing content="" in that case is incorrect and will confuse
        the model on the follow-up turn.
        """
        msg: ChatCompletionAssistantMessageParam = {"role": "assistant"}
        if content is not None:
            msg["content"] = content
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls  # type: ignore[typeddict-unknown-key]
        self._append(msg)

    def tool(self, tool_call_id: str, result: str) -> None:
        """Inject a tool result back into the conversation."""
        msg: ChatCompletionToolMessageParam = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        }
        self._append(msg)

    # ------------------------------------------------------------------
    # Context window management
    # ------------------------------------------------------------------

    def token_count(self) -> int:
        """Total tokens across all messages — O(1), tracked incrementally."""
        return self._total_tokens

    def _next_group_size(self) -> int:
        """Number of messages in the next atomic drop group.

        An assistant message that carries tool_calls must be dropped together
        with all immediately following tool-result messages, otherwise the
        model sees an orphaned tool result and either errors or hallucinates.
        Every other message (user, plain assistant) is a group of one.
        """
        if not self._turns:
            return 0
        first = self._turns[0]
        if first.get("role") == "assistant" and first.get("tool_calls"):
            size = 1
            for msg in itertools.islice(self._turns, 1, None):
                if msg.get("role") == "tool":
                    size += 1
                else:
                    break
            return size
        return 1

    def trim_to_budget(self, max_tokens: int) -> None:
        """Drop oldest turn-groups until the total is within budget.

        System message is always preserved.  Groups are dropped atomically so
        an assistant[tool_calls] message is never separated from its tool
        results.  Running total is updated per-drop so the check is O(1).
        Uses deque.popleft() for O(1) removal.
        """
        before = self._total_tokens
        dropped = 0

        while self._turns and self._total_tokens > max_tokens:
            group_size = self._next_group_size()
            for _ in range(group_size):
                removed = self._turns.popleft()
                self._total_tokens -= self._msg_tokens(removed)
                dropped += 1

        if dropped:
            logger.debug(
                "trim: dropped %d turn(s), %d → %d tokens",
                dropped,
                before,
                self._total_tokens,
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_messages(self) -> list[ChatCompletionMessageParam]:
        """Return the full message list ready to pass to the OpenAI API."""
        return [self._system, *self._turns]

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._turns)

    def __repr__(self) -> str:
        return f"Chat(turns={len(self._turns)}, tokens={self._total_tokens})"
