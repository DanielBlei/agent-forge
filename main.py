import asyncio
import logging
import select
import sys

from agent_forge.agent import Agent
from agent_forge.cli import parse_args
from agent_forge.client import ClientConfig, format_model_error_help, list_available_models
from agent_forge.config import load_config
from agent_forge.logger import get_logger, set_log_level

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
    logger.info("agent-forge starting")
    try:
        config = load_config(args.config)

        if args.debug:
            set_log_level(logging.DEBUG)

        async with await Agent.from_config(config) as agent:
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

                await agent.run(user_input)

                if logger.isEnabledFor(logging.DEBUG):
                    used = agent.chat.token_count()
                    logger.debug(
                        "[ctx] %d / %d tokens | %d left | %d turns",
                        used, agent.cfg.context_limit, agent.cfg.context_limit - used, len(agent.chat),
                    )

    except Exception as e:  # noqa: BLE001
        logger.error("runner failure: %s", e)
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ["does not exist", "not found", "404", "no such model"]):
            logger.info("Model not found — fetching available models...")
            cfg = ClientConfig(
                model=config.model.name,
                base_url=config.model.endpoint,
                context_limit=config.model.context_limit,
            )
            available_models = await list_available_models(cfg)
            logger.error("\n%s", format_model_error_help(cfg, available_models))
        sys.exit(1)


if __name__ == "__main__":
    run()
