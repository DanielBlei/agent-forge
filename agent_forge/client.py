from dataclasses import dataclass

from openai import AsyncOpenAI


@dataclass
class ClientConfig:
    base_url: str
    model: str
    context_limit: int
    api_key: str = "local"


def new_client(cfg: ClientConfig) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
