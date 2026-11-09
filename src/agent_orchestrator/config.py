"""Declarative configuration (``agents.yaml``)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

_ENV = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


class ProviderConfig(BaseModel):
    type: Literal["openai", "anthropic"]
    base_url: str | None = None
    api_key: str | None = None


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
    provider: str
    model: str
    instructions: str
    tools: list[str] = Field(default_factory=list)  # builtin names or "<mcp_server>__<tool>" or "<mcp_server>__*"
    delegates: list[str] = Field(default_factory=list)  # agents this one can hand work to
    max_steps: int = Field(default=8, ge=1, le=50)
    max_tokens: int = Field(default=1024, ge=16)
    temperature: float = Field(default=0.2, ge=0, le=2)


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
            if a.provider not in self.providers:
                raise ValueError(f"agent {a.name!r} uses unknown provider {a.provider!r}")
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
