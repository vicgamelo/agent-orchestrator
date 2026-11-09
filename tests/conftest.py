from typing import Any

import pytest

from agent_orchestrator.config import AppConfig


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def make_config(**overrides: Any) -> AppConfig:
    """Coordinator delegating to two workers, all on the scripted provider."""
    base: dict[str, Any] = {
        "providers": {"fake": {"type": "openai", "base_url": "http://unused"}},
        "agents": [
            {
                "name": "coordinator",
                "description": "boss",
                "provider": "fake",
                "model": "coordinator",
                "instructions": "coordinate",
                "delegates": ["economist", "researcher"],
            },
            {
                "name": "economist",
                "description": "numbers",
                "provider": "fake",
                "model": "economist",
                "instructions": "econ",
                "tools": ["calculator"],
            },
            {
                "name": "researcher",
                "description": "facts",
                "provider": "fake",
                "model": "researcher",
                "instructions": "r",
            },
        ],
        "default_agent": "coordinator",
    }
    base.update(overrides)
    return AppConfig.model_validate(base)
