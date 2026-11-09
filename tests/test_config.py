from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_orchestrator.config import load_config

from .conftest import make_config


def test_example_config_loads_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_MODEL", "llama3.2")
    monkeypatch.delenv("MCP_BRASIL_URL", raising=False)
    cfg = load_config(Path(__file__).parent.parent / "agents.yaml")
    assert cfg.default_agent == "coordinator"
    assert {a.name: a.model for a in cfg.agents}["economist"] == "llama3.2"
    assert cfg.mcp_servers[0].url == "https://mcp.vgmelo.tech/mcp"


def test_rejects_unknown_delegate() -> None:
    with pytest.raises(ValidationError, match="invalid delegate"):
        make_config(
            agents=[
                {
                    "name": "a",
                    "description": "d",
                    "provider": "fake",
                    "model": "m",
                    "instructions": "i",
                    "delegates": ["ghost"],
                }
            ],
            default_agent="a",
        )


def test_mcp_server_needs_exactly_one_transport() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        make_config(mcp_servers=[{"name": "x", "url": "http://a", "command": "uvx"}])
