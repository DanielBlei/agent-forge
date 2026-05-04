from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from jsonschema import validate
from jsonschema.exceptions import ValidationError

from agent_forge.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    name: str
    endpoint: str
    context_limit: int


@dataclass
class SystemConfig:
    prompt: str


@dataclass
class RagConfig:
    endpoint: str  # host:port gRPC address, e.g. "localhost:50051"


@dataclass
class AppConfig:
    model: ModelConfig
    system: SystemConfig
    rag: RagConfig | None = None


def _load_schema() -> dict:
    """Load JSON schema for configuration validation."""
    schema_path = Path(__file__).parent.parent / "config.schema.yaml"
    with schema_path.open() as f:
        return yaml.safe_load(f)


def _apply_env_overrides(config: AppConfig) -> AppConfig:
    """Apply environment variable overrides to configuration."""
    model = config.model
    if endpoint := os.environ.get("AGENT_FORGE_ENDPOINT"):
        logger.debug("Overrode endpoint with AGENT_FORGE_ENDPOINT: %s", endpoint)
        model = replace(model, endpoint=endpoint)
    if model_name := os.environ.get("OLLAMA_MODEL"):
        logger.debug("Overrode model name with OLLAMA_MODEL: %s", model_name)
        model = replace(model, name=model_name)
    return replace(config, model=model)


def load_config(config_path: str) -> AppConfig:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated AppConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config doesn't match schema
        ValueError: If required sections are missing
    """
    try:
        # Load YAML file
        with Path(config_path).open() as f:
            config_data = yaml.safe_load(f)

        if config_data is None:
            raise ValueError("Configuration file is empty")

        # Validate against schema
        schema = _load_schema()
        validate(instance=config_data, schema=schema)

        # Create configuration objects
        rag: RagConfig | None = None
        if rag_data := config_data.get("rag"):
            rag = RagConfig(endpoint=rag_data["endpoint"])

        config = AppConfig(
            model=ModelConfig(
                name=config_data["model"]["name"],
                endpoint=config_data["model"]["endpoint"],
                context_limit=config_data["model"]["context_limit"],
            ),
            system=SystemConfig(
                prompt=config_data["system"]["prompt"],
            ),
            rag=rag,
        )

        # Apply environment variable overrides
        config = _apply_env_overrides(config)

        logger.debug("Configuration loaded successfully from %s", config_path)
        return config

    except FileNotFoundError:
        logger.error("Configuration file not found: %s", config_path)
        raise
    except ValidationError as e:
        logger.error("Configuration validation failed: %s", e.message)
        raise ValueError(f"Invalid configuration: {e.message}") from e
    except (yaml.YAMLError, KeyError) as e:
        logger.error("Configuration file format error: %s", e)
        raise ValueError(f"Configuration file error: {e}") from e
