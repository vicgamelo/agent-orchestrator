"""Declarative configuration (``agents.yaml``)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ENV = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


class ProviderConfig(BaseModel):
    type: Literal["openai", "anthropic"]
    base_url: str | None = None
    api_key: str | None = None
    timeout_s: float = Field(default=120.0, gt=0)
    # Refuse to start with a model that isn't ``...:free`` (OpenRouter bills the others against credits).
    free_models_only: bool = False
    # Sent as ``reasoning_effort`` (Groq / OpenAI reasoning models). "low" keeps gpt-oss from spending
    # the whole max_tokens thinking and answering nothing.
    reasoning_effort: Literal["low", "medium", "high"] | None = None


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


class StudioModel(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", max_length=40)  # what callers send
    label: str
    provider: str
    model: str
    enabled: bool = True  # e.g. ${STUDIO_LOCAL:-false} for an Ollama model only on a dev machine


class StudioConfig(BaseModel):
    """Agents written by API callers (the agent studio): one agent, no delegation, only these
    models and tools, and tighter limits than the configured team."""

    models: list[StudioModel] = Field(min_length=1)
    tools: list[str] = Field(default_factory=list)  # same refs as an agent's `tools`
    templates: list[str] = Field(default_factory=list)  # configured agents offered as starting points
    max_steps: int = Field(default=6, ge=1, le=20)
    max_tokens: int = Field(default=2048, ge=16)
    instructions_max_chars: int = Field(default=4000, ge=100)


class StudioAgent(BaseModel):
    """An agent sent with the request. The orchestrator checks it against ``StudioConfig``."""

    model_config = ConfigDict(extra="forbid")
    instructions: str = Field(min_length=1, max_length=20_000)
    model: str = Field(max_length=40)  # a StudioModel id
    tools: list[str] = Field(default_factory=list, max_length=40)
    temperature: float = Field(default=0.2, ge=0, le=1.5)
    max_steps: int = Field(default=4, ge=1, le=20)


class AppConfig(BaseModel):
    providers: dict[str, ProviderConfig]
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    agents: list[AgentConfig]
    default_agent: str
    limits: Limits = Field(default_factory=Limits)
    studio: StudioConfig | None = None

    @model_validator(mode="after")
    def _check_references(self) -> AppConfig:
        names = {a.name for a in self.agents}
        if self.default_agent not in names:
            raise ValueError(f"default_agent {self.default_agent!r} is not defined")
        if self.studio:
            ids = [m.id for m in self.studio.models]
            if len(set(ids)) != len(ids):
                raise ValueError("studio: repeated model ids")
            for m in self.studio.models:
                if m.provider not in self.providers:
                    raise ValueError(f"studio model {m.id!r} uses unknown provider {m.provider!r}")
                if self.providers[m.provider].free_models_only and not m.model.endswith(":free"):
                    raise ValueError(f"studio model {m.id!r}: provider {m.provider!r} only takes ':free' models")
            for t in self.studio.templates:
                if t not in names:
                    raise ValueError(f"studio template {t!r} is not a configured agent")
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
