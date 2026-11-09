"""Declarative configuration (``agents.yaml``)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

_ENV = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


class ProviderConfig(BaseModel):
    type: Literal["openai", "anthropic"]
    base_url: str | None = None
    api_key: str | None = None
    timeout_s: float = Field(default=120.0, gt=0)
    # Refuse to start with a model that isn't ``...:free`` (OpenRouter bills the others against credits).
    free_models_only: bool = False


class MCPServerConfig(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_transport(self) -> MCPServerConfig:
        if bool(self.url) == bool(self.command):
            raise ValueError(f"mcp server {self.name!r}: set exactly one of `url` or `command`")
        return self


class AgentConfig(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str
    # One provider, or an ordered fallback chain: [groq, openrouter] (or "groq,openrouter" from an env var).
    provider: str | list[str]
    model: str | None = None
    models: dict[str, str] = Field(default_factory=dict)  # per-provider model, wins over `model`
    instructions: str
    tools: list[str] = Field(default_factory=list)  # builtin names or "<mcp_server>__<tool>" or "<mcp_server>__*"
    delegates: list[str] = Field(default_factory=list)  # agents this one can hand work to
    max_steps: int = Field(default=8, ge=1, le=50)
    max_tokens: int = Field(default=1024, ge=16)
    temperature: float = Field(default=0.2, ge=0, le=2)

    @field_validator("provider", mode="before")
    @classmethod
    def _split_provider(cls, v: object) -> object:
        if isinstance(v, str) and "," in v:
            return [p.strip() for p in v.split(",") if p.strip()]
        return v

    @property
    def provider_chain(self) -> list[str]:
        return [self.provider] if isinstance(self.provider, str) else list(self.provider)

    def model_for(self, provider: str) -> str:
        model = self.models.get(provider) or self.model
        if not model:
            raise ValueError(f"agent {self.name!r} has no model for provider {provider!r}")
        return model


class Limits(BaseModel):
    tool_timeout_s: float = 30.0
    run_timeout_s: float = 180.0
    max_delegation_depth: int = 2
    max_total_tokens: int = 200_000


class AppConfig(BaseModel):
    providers: dict[str, ProviderConfig]
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    agents: list[AgentConfig]
    default_agent: str
    limits: Limits = Field(default_factory=Limits)

    @model_validator(mode="after")
    def _check_references(self) -> AppConfig:
        names = {a.name for a in self.agents}
        if self.default_agent not in names:
            raise ValueError(f"default_agent {self.default_agent!r} is not defined")
        for a in self.agents:
            chain = a.provider_chain
            if not chain or len(set(chain)) != len(chain):
                raise ValueError(f"agent {a.name!r}: provider list must be non-empty and without repeats")
            for p in chain:
                if p not in self.providers:
                    raise ValueError(f"agent {a.name!r} uses unknown provider {p!r}")
                model = a.model_for(p)
                if self.providers[p].free_models_only and not model.endswith(":free"):
                    raise ValueError(f"agent {a.name!r}: provider {p!r} only takes ':free' models, got {model!r}")
            for d in a.delegates:
                if d not in names or d == a.name:
                    raise ValueError(f"agent {a.name!r} has invalid delegate {d!r}")
        return self


def _expand_env(text: str) -> str:
    return _ENV.sub(lambda m: os.getenv(m.group(1), m.group(2) or ""), text)


def load_config(path: str | Path) -> AppConfig:
    """Load YAML, expanding ``${VAR}`` and ``${VAR:-default}`` from the environment."""
    raw = yaml.safe_load(_expand_env(Path(path).read_text()))
    return AppConfig.model_validate(raw)
