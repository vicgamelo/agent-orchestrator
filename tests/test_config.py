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


def _agent(**fields: object) -> dict[str, object]:
    return {"name": "a", "description": "d", "instructions": "i", **fields}


def _providers() -> dict[str, object]:
    return {"groq": {"type": "openai"}, "openrouter": {"type": "openai"}, "local": {"type": "openai"}}


def test_provider_accepts_a_fallback_chain() -> None:
    cfg = make_config(
        providers=_providers(),
        agents=[_agent(provider=["groq", "openrouter"], model="m", models={"openrouter": "x:free"})],
        default_agent="a",
    )
    a = cfg.agents[0]
    assert a.provider_chain == ["groq", "openrouter"]
    assert (a.model_for("groq"), a.model_for("openrouter")) == ("m", "x:free")


def test_provider_chain_from_comma_separated_env_value() -> None:
    cfg = make_config(
        providers=_providers(), agents=[_agent(provider="groq, openrouter", model="m")], default_agent="a"
    )
    assert cfg.agents[0].provider_chain == ["groq", "openrouter"]


def test_single_provider_still_a_string() -> None:
    cfg = make_config(providers=_providers(), agents=[_agent(provider="local", model="m")], default_agent="a")
    assert cfg.agents[0].provider == "local" and cfg.agents[0].provider_chain == ["local"]


def test_rejects_unknown_provider_in_chain() -> None:
    with pytest.raises(ValidationError, match="unknown provider 'grok'"):
        make_config(
            providers=_providers(), agents=[_agent(provider=["grok", "openrouter"], model="m")], default_agent="a"
        )


def test_rejects_chain_without_model_for_a_provider() -> None:
    with pytest.raises(ValidationError, match="no model for provider 'openrouter'"):
        make_config(
            providers=_providers(),
            agents=[_agent(provider=["groq", "openrouter"], models={"groq": "g"})],
            default_agent="a",
        )


def test_rejects_repeated_provider_in_chain() -> None:
    with pytest.raises(ValidationError, match="without repeats"):
        make_config(providers=_providers(), agents=[_agent(provider=["groq", "groq"], model="m")], default_agent="a")


def test_example_config_free_tier_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "GROQ_COORDINATOR_MODEL",
        "OPENROUTER_COORDINATOR_MODEL",
        "GROQ_WORKER_MODEL",
        "OPENROUTER_WORKER_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("COORDINATOR_PROVIDER", "groq,openrouter")
    monkeypatch.setenv("WORKER_PROVIDER", "[groq, openrouter]")  # YAML list syntax works too
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    cfg = load_config(Path(__file__).parent.parent / "agents.yaml")
    agents = {a.name: a for a in cfg.agents}
    coord, econ = agents["coordinator"], agents["economist"]
    assert coord.provider_chain == econ.provider_chain == ["groq", "openrouter"]
    assert coord.model_for("groq") == "openai/gpt-oss-120b"
    assert coord.model_for("openrouter") == "google/gemma-4-31b-it:free"
    assert econ.model_for("groq") == "openai/gpt-oss-20b"
    assert cfg.providers["groq"].base_url == "https://api.groq.com/openai/v1"
    assert cfg.providers["groq"].api_key == "gsk_test"
    assert cfg.providers["openrouter"].base_url == "https://openrouter.ai/api/v1"


def test_free_only_provider_rejects_a_paid_model() -> None:
    # Um id sem ":free" no OpenRouter gasta crédito: a config nem sobe.
    providers = {**_providers(), "openrouter": {"type": "openai", "base_url": "u", "free_models_only": True}}
    with pytest.raises(ValidationError, match="only takes ':free' models, got 'google/gemma-4-31b-it'"):
        make_config(
            providers=providers,
            agents=[_agent(provider=["groq", "openrouter"], model="m", models={"openrouter": "google/gemma-4-31b-it"})],
            default_agent="a",
        )
    ok = make_config(
        providers=providers,
        agents=[_agent(provider=["groq", "openrouter"], model="m", models={"openrouter": "x:free"})],
        default_agent="a",
    )
    assert ok.agents[0].model_for("openrouter") == "x:free"


def test_example_config_refuses_paid_openrouter_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_PROVIDER", "groq,openrouter")
    monkeypatch.setenv("OPENROUTER_COORDINATOR_MODEL", "anthropic/claude-sonnet-5-5")
    with pytest.raises(ValidationError, match="only takes ':free' models"):
        load_config(Path(__file__).parent.parent / "agents.yaml")
