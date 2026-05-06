from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from typing import Any

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, BadRequestError
from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionToolParam
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent_forge.chat import Chat
from agent_forge.client import ClientConfig
from agent_forge.logger import get_logger

_MODEL_PREFIX = "\033[32mmodel: \033[0m"  # green

logger = get_logger(__name__)

ToolHandler = Callable[..., Any]

_TRANSIENT_ERRORS = (APIConnectionError, APITimeoutError)


def _is_context_error(e: BadRequestError) -> bool:
    # SDK exposes a `code` field for structured errors; fall back to string match
    # only for the specific context-length phrase to avoid matching auth/token errors.
    code = getattr(e, "code", None) or ""
    if code in ("context_length_exceeded", "max_tokens_exceeded"):
        return True
    msg = str(e).lower()
    return "context length" in msg or "context_length_exceeded" in msg


def _trim_context(chat: Chat, cfg: ClientConfig) -> bool:
    """Trim context and return True if trimming occurred."""
    before = chat.token_count()
    chat.trim_to_budget(cfg.context_limit)
    return chat.token_count() < before


async def _make_api_call(
    client: AsyncOpenAI,
    chat: Chat,
    cfg: ClientConfig,
    tools: list[ChatCompletionToolParam] | None,
) -> Any:
    """Make API call with retry logic for transient errors."""
    kwargs: dict[str, Any] = {}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    async for attempt in AsyncRetrying(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        reraise=True,
    ):
        with attempt:
            return await client.chat.completions.create(
                model=cfg.model,
                messages=chat.to_messages(),
                stream=True,
                **kwargs,
            )


def _apply_tool_call(accum: dict[int, dict[str, Any]], tc_delta: Any) -> None:
    idx = tc_delta.index
    if idx not in accum:
        accum[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
    if tc_delta.id:
        accum[idx]["id"] = tc_delta.id
    if tc_delta.function:
        if tc_delta.function.name:
            accum[idx]["function"]["name"] += tc_delta.function.name
        if tc_delta.function.arguments:
            accum[idx]["function"]["arguments"] += tc_delta.function.arguments


async def _process_stream(
    stream: Any,
) -> tuple[str, str | None, dict[int, dict[str, Any]]]:
    """Process streaming response, accumulate content and tool calls.

    Returns tuple of (text_content, finish_reason, tool_call_accumulator).
    """
    content_parts: list[str] = []
    tool_call_accum: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    header_printed = False

    async for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        finish_reason = choice.finish_reason or finish_reason
        delta = choice.delta

        if delta.content:
            if not header_printed:
                sys.stdout.write(_MODEL_PREFIX)
                header_printed = True
            sys.stdout.write(delta.content)
            sys.stdout.flush()
            content_parts.append(delta.content)

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                _apply_tool_call(tool_call_accum, tc_delta)

    if header_printed:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return "".join(content_parts), finish_reason, tool_call_accum


async def _run_single_tool(
    tc: ChatCompletionMessageToolCall,
    tool_handlers: dict[str, ToolHandler] | None,
) -> tuple[str, str]:
    """Run one tool call and return (tool_call_id, result_json)."""
    fn_name = tc["function"]["name"]
    fn_args: dict[str, Any] = json.loads(tc["function"]["arguments"])
    logger.debug("tool call: %s(%s)", fn_name, fn_args)

    handler = (tool_handlers or {}).get(fn_name)
    if handler is None:
        logger.warning("no handler for tool %r", fn_name)
        return tc["id"], f"Error: no handler registered for tool '{fn_name}'"

    try:
        if asyncio.iscoroutinefunction(handler):
            ret = await handler(**fn_args)
        else:
            # Sync handler — run in thread pool to avoid blocking the event loop
            ret = await asyncio.to_thread(handler, **fn_args)
        result = json.dumps(ret, default=str)
        logger.debug("tool %r → %d chars", fn_name, len(result))
        return tc["id"], result
    except Exception as e:
        logger.exception("tool %r raised", fn_name)
        return tc["id"], f"Error: {e}"


async def _execute_tool_calls(
    tool_calls: list[ChatCompletionMessageToolCall],
    tool_handlers: dict[str, ToolHandler] | None,
    chat: Chat,
) -> None:
    """Execute all tool calls in parallel and inject results into chat in order."""
    results = await asyncio.gather(*[_run_single_tool(tc, tool_handlers) for tc in tool_calls])
    for tool_call_id, result in results:
        chat.tool(tool_call_id, result)


async def agent_loop(
    client: AsyncOpenAI,
    chat: Chat,
    cfg: ClientConfig,
    tools: list[ChatCompletionToolParam] | None = None,
    tool_handlers: dict[str, ToolHandler] | None = None,
) -> str:
    """Send messages and handle tool-call rounds until the model returns a text response.

    Streams the final text response to stdout as tokens arrive.  Tool-call
    rounds execute silently and loop back.  Context is trimmed atomically
    (whole tool-call groups) before each API call.

    Transient errors are retried with exponential backoff (3 attempts).
    A 400 context-length error triggers an aggressive re-trim and one retry.
    """
    compact_announced = False
    context_error_retried = False
    round_num = 0

    while True:
        round_num += 1
        logger.debug("[round %d] starting — %d turns, %d tokens", round_num, len(chat), chat.token_count())

        if _trim_context(chat, cfg) and not compact_announced:
            logger.info("auto-compact: context limit reached — trimming oldest turns as we go")
            compact_announced = True

        try:
            stream = await _make_api_call(client, chat, cfg, tools)
            context_error_retried = False  # reset after a successful call
        except BadRequestError as e:
            if not context_error_retried and _is_context_error(e):
                logger.warning("context error from API — force-trimming to 75%% of limit and retrying")
                chat.trim_to_budget(int(cfg.context_limit * 0.75))
                context_error_retried = True
                continue
            raise

        text, finish_reason, tool_call_accum = await _process_stream(stream)
        logger.debug(
            "[round %d] finish_reason=%r tool_calls_in_stream=%d", round_num, finish_reason, len(tool_call_accum)
        )

        if tool_call_accum and (finish_reason in ("tool_calls", "stop")):
            tool_calls = [tool_call_accum[i] for i in sorted(tool_call_accum)]
            logger.debug(
                "[round %d] dispatching %d tool call(s): %s",
                round_num,
                len(tool_calls),
                [tc["function"]["name"] for tc in tool_calls],
            )
            chat.assistant(content=text or None, tool_calls=tool_calls)
            await _execute_tool_calls(tool_calls, tool_handlers, chat)
            continue

        if not text.strip() and not tool_call_accum:
            logger.warning(
                "[round %d] model stalled: finish_reason=%r, no text and no tool calls — "
                "check model capability or tool_choice setting",
                round_num,
                finish_reason,
            )
            return ""

        logger.debug("[round %d] final text response (%d chars)", round_num, len(text))
        chat.assistant(text)
        return text
