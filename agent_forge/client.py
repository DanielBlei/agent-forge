import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    base_url: str
    model: str
    context_limit: int
    api_key: str = "local"


def new_client(cfg: ClientConfig) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)


async def list_available_models(cfg: ClientConfig) -> list[str]:
    """Fetch available models from the configured API endpoint.

    Returns an empty list if the endpoint is unreachable or returns an error.
    """
    try:
        client = new_client(cfg)
        response = await client.models.list()
        return [model.id for model in response.data]
    except Exception as e:
        logger.debug("could not fetch model list: %s", e)
        return []


def format_model_error_help(cfg: ClientConfig, available_models: list[str]) -> str:
    """Format a user-facing help message for model configuration issues."""
    lines = [
        f"Model '{cfg.model}' not found at {cfg.base_url}",
        "Available models:"
    ]
    
    if available_models:
        for model in available_models:
            lines.append(f"  • {model}")
        lines.append("\nPlease update your config file to use one of the available models shown above.")
    else:
        lines.append("  [Could not retrieve model list]")
        lines.append("\nPlease check:")
        lines.append("  • LLM service is running at the specified endpoint")
        lines.append("  • Endpoint URL includes '/v1' (e.g., 'http://localhost:8000/v1')")
        lines.append("  • Correct port number is configured")
    
    return "\n".join(lines)
