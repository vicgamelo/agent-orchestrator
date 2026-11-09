"""Expose tools from MCP servers to agents.

Each configured server gets one long-lived MCP client session (stdio subprocess
or Streamable HTTP). Its tools are namespaced as ``<server>__<tool>`` so two
servers can expose tools with the same name.
"""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp.client import Client

from .config import MCPServerConfig
from .tools import Tool

logger = logging.getLogger(__name__)

MAX_RESULT_CHARS = 20_000


class MCPToolset:
    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self.clients: dict[str, Client] = {}
        self.tools: dict[str, Tool] = {}

    async def connect(self, name: str, target: Any) -> list[Tool]:
        """``target`` is anything ``mcp.client.Client`` accepts: a URL, stdio
        parameters, or an in-process server (handy for tests)."""
        client = await self._stack.enter_async_context(Client(target))
        self.clients[name] = client
        listed = await client.list_tools()
        tools = [self._wrap(name, client, t) for t in listed.tools]
        for t in tools:
            self.tools[t.name] = t
        logger.info("mcp server %s: %d tools", name, len(tools))
        return tools

    async def connect_config(self, cfg: MCPServerConfig) -> list[Tool]:
        if cfg.url:
            return await self.connect(cfg.name, cfg.url)
        from mcp.client.stdio import StdioServerParameters

        params = StdioServerParameters(command=cfg.command or "", args=cfg.args, env=cfg.env or None)
        return await self.connect(cfg.name, params)

    async def aclose(self) -> None:
        await self._stack.aclose()

    @staticmethod
    def _wrap(server: str, client: Client, tool: Any) -> Tool:
        async def call(arguments: dict[str, Any]) -> str:
            result = await client.call_tool(tool.name, arguments)
            if result.structured_content is not None:
                text = json.dumps(result.structured_content, ensure_ascii=False)
            else:
                text = "\n".join(getattr(c, "text", "") for c in result.content)
            if result.is_error:
                text = f"error: {text}"
            return text[:MAX_RESULT_CHARS]

        return Tool(
            name=f"{server}__{tool.name}",
            description=tool.description or tool.name,
            parameters=tool.input_schema or {"type": "object", "properties": {}},
            fn=call,
            source=f"mcp:{server}",
        )
