from __future__ import annotations

from typing import Self

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionToolParam

from agent_forge.chat import Chat
from agent_forge.client import ClientConfig, new_client
from agent_forge.config import AppConfig
from agent_forge.loop import ToolHandler, agent_loop
from agent_forge.rag_tools import init_rag_tools
from agent_forge.tools import TOOL_DEFINITIONS, TOOL_HANDLERS


def _build_system_prompt(base: str, tools: list[ChatCompletionToolParam]) -> str:
    """Prepend a tool-awareness preamble derived from registered tools."""
    if not tools:
        return base
    names = ", ".join(t["function"]["name"] for t in tools)
    preamble = (
        f"You have access to these tools: {names}. "
        "Use them proactively — do not ask the user for information you can retrieve yourself. "
        "When you know a filename (e.g. 'main.py', 'client.py'), call read_file directly with that name. "
        "Use glob_files only when you need to discover files by pattern. "
        "Each tool call handles one file or pattern — make multiple calls for multiple files.\n\n"
    )
    return preamble + base


class Agent:
    """Encapsulates the full agent lifecycle: client, chat, tools, and RAG channel."""

    def __init__(
        self,
        client: AsyncOpenAI,
        chat: Chat,
        cfg: ClientConfig,
        tools: list[ChatCompletionToolParam],
        tool_handlers: dict[str, ToolHandler],
        _rag_channel=None,
    ) -> None:
        self.client = client
        self.chat = chat
        self.cfg = cfg
        self.tools = tools
        self.tool_handlers = tool_handlers
        self._rag_channel = _rag_channel

    @classmethod
    async def from_config(cls, config: AppConfig) -> Agent:
        """Construct an Agent from an AppConfig, wiring all parts internally."""
        cfg = ClientConfig(
            model=config.model.name,
            base_url=config.model.endpoint,
            context_limit=config.model.context_limit,
        )

        tool_defs = list(TOOL_DEFINITIONS)
        tool_handlers: dict[str, ToolHandler] = dict(TOOL_HANDLERS)
        rag_channel = None

        if config.rag:
            rag_defs, rag_hdlrs, rag_channel = init_rag_tools(config.rag.endpoint)
            tool_defs.extend(rag_defs)
            tool_handlers.update(rag_hdlrs)

        system = _build_system_prompt(config.system.prompt, tool_defs)
        chat = Chat(system=system)
        client = new_client(cfg)
        return cls(client, chat, cfg, tool_defs, tool_handlers, rag_channel)

    async def run(self, user_input: str) -> str:
        """Append a user turn and run one agent loop iteration."""
        self.chat.user(user_input)
        return await agent_loop(
            client=self.client,
            chat=self.chat,
            cfg=self.cfg,
            tools=self.tools,
            tool_handlers=self.tool_handlers,
        )

    async def close(self) -> None:
        if self._rag_channel:
            await self._rag_channel.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()
