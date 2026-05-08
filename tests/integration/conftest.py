"""
Integration test fixtures: config loading, endpoint reachability check,
per-test Agent creation, and session-scoped benchmark result collection.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio
from openai import AsyncOpenAI

from agent_forge.agent import Agent
from agent_forge.config import AppConfig, load_config

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.yaml"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--config",
        default=None,
        help="Path to agent-forge config YAML (overrides AGENT_FORGE_CONFIG env var)",
    )


def _load_app_config(config: pytest.Config) -> AppConfig:
    path = (
        config.getoption("--config")
        or os.environ.get("AGENT_FORGE_CONFIG")
        or str(_DEFAULT_CONFIG)
    )
    return load_config(path)


async def _reachable(endpoint: str) -> bool:
    try:
        client = AsyncOpenAI(base_url=endpoint, api_key="local")
        await asyncio.wait_for(client.models.list(), timeout=5.0)
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def app_config(request: pytest.FixtureRequest) -> AppConfig:
    try:
        cfg = _load_app_config(request.config)
    except FileNotFoundError:
        pytest.skip("no config.yaml — pass --config or set AGENT_FORGE_CONFIG")
    if not asyncio.run(_reachable(cfg.model.endpoint)):
        pytest.skip(f"model endpoint unreachable: {cfg.model.endpoint}")
    return cfg


@pytest_asyncio.fixture
async def agent(app_config: AppConfig) -> Agent:
    async with await Agent.from_config(app_config) as a:
        yield a


@pytest.fixture(scope="session")
def benchmark_results(app_config: AppConfig) -> list:
    results: list[dict] = []
    yield results

    if not results:
        return

    results_dir = _PROJECT_ROOT / "tests" / "benchmarks"
    results_dir.mkdir(exist_ok=True)

    model_safe = app_config.model.name.replace("/", "-").replace(":", "-")
    ts = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M")
    out = results_dir / f"{ts}_{model_safe}.json"

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    tool_hit = sum(1 for r in results if r["tool_called"])

    categories: dict[str, dict] = {}
    for r in results:
        cat = r.get("category", "uncategorized")
        bucket = categories.setdefault(cat, {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += int(r["passed"])
    by_category = {
        cat: {"total": v["total"], "passed": v["passed"], "pass_rate": round(v["passed"] / v["total"], 3)}
        for cat, v in sorted(categories.items())
    }

    payload = {
        "model": app_config.model.name,
        "endpoint": app_config.model.endpoint,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 3),
            "tool_call_rate": round(tool_hit / total, 3),
            "by_category": by_category,
        },
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n\nbenchmark → {out}")
