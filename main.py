import asyncio
import logging
import select
import sys

from agent_forge.chat import Chat
from agent_forge.cli import parse_args
from agent_forge.client import ClientConfig, new_client
from agent_forge.config import load_config
from agent_forge.logger import get_logger, set_log_level
from agent_forge.loop import agent_loop
from agent_forge.rag_tools import init_rag_tools
from agent_forge.tools import TOOL_DEFINITIONS, TOOL_HANDLERS

_USER_PROMPT = "\033[36muser: \033[0m"  # cyan

logger = get_logger(__name__)


def _read_input() -> str:
    """Read one logical message from stdin.

    Typing a single line and pressing Enter submits immediately — no extra
    keypress needed.  Pasting multi-line content (code, context, etc.) works
    because all pasted lines arrive buffered together; we drain stdin greedily
    with a short timeout so blank lines inside the paste are preserved.

    Uses sys.stdin.readline() throughout to avoid buffer-mixing issues with
    input(), which would leave a stray newline and cause a double prompt.
    """
    sys.stdout.write(_USER_PROMPT)
    sys.stdout.flush()
    first = sys.stdin.readline()
    if not first:  # EOF
        raise EOFError
    first = first.rstrip("\n")
    if not first.strip():
        return ""

    lines = [first]
    while select.select([sys.stdin], [], [], 0.05)[0]:
        line = sys.stdin.readline()
        if not line:  # EOF mid-paste
            break
        lines.append(line.rstrip("\n"))

    return "\n".join(lines).strip()


def run():
    args = parse_args()
    asyncio.run(runner(args))


async def runner(args) -> None:
    rag_channel = None
    try:
        logger.info("agent-forge starting")

        config = load_config(args.config)

        if args.debug:
            set_log_level(logging.DEBUG)

        cfg = ClientConfig(
            model=config.model.name,
            base_url=config.model.endpoint,
            context_limit=config.model.context_limit,
        )

        tool_defs = list(TOOL_DEFINITIONS)
        tool_handlers = dict(TOOL_HANDLERS)

        if config.rag:
            rag_defs, rag_handlers, rag_channel = init_rag_tools(config.rag.endpoint)
            tool_defs.extend(rag_defs)
            tool_handlers.update(rag_handlers)

        chat = Chat(system=config.system.prompt)
        client = new_client(cfg=cfg)

        while True:
            try:
                user_input = _read_input()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue

            if user_input.lower() in {"/exit", "/quit", "/bye"}:
                logger.info("exiting, see you next time!")
                break

            chat.user(user_input)
            await agent_loop(
                client=client,
                chat=chat,
                cfg=cfg,
                tools=tool_defs,
                tool_handlers=tool_handlers,
            )
            if logger.isEnabledFor(logging.DEBUG):
                used = chat.token_count()
                logger.debug(
                    "[ctx] %d / %d tokens | %d left | %d turns",
                    used, cfg.context_limit, cfg.context_limit - used, len(chat),
                )

    except Exception as e:  # noqa: BLE001
        logger.error("runner failure: %s", e)
        sys.exit(1)
    finally:
        if rag_channel:
            await rag_channel.close()


if __name__ == "__main__":
    run()
