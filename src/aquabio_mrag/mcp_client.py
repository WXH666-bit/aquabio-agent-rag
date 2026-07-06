from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.client.stdio import stdio_client


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    command: str
    args: list[str]
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 180.0
    retries: int = 1
    url: str = ""


class MCPStdioClient:
    """Real MCP stdio client with initialize, discovery and tool calls."""

    def __init__(self, servers: dict[str, MCPServerConfig]):
        self.servers = servers

    @staticmethod
    def _decode_result(value: Any) -> Any:
        if (
            isinstance(value, dict)
            and set(value) == {"result"}
        ):
            value = value["result"]
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    async def _session_call(
        self,
        server: str,
        operation: str,
        tool_name: str = "",
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        config = self.servers[server]
        env = os.environ.copy()
        env.update(config.env)
        if config.url:
            transport = streamable_http_client(
                config.url,
                terminate_on_close=False,
            )
        else:
            params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=env,
            )
            transport = stdio_client(params)
        decoded: Any
        async with transport as streams:
            reader, writer = streams[0], streams[1]
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                if operation == "list":
                    result = await session.list_tools()
                    decoded = [
                        {
                            "name": tool.name,
                            "description": tool.description or "",
                            "input_schema": tool.inputSchema,
                        }
                        for tool in result.tools
                    ]
                else:
                    result = await session.call_tool(
                        tool_name, arguments=arguments or {}
                    )
                    if (
                        getattr(result, "structuredContent", None)
                        is not None
                    ):
                        decoded = self._decode_result(
                            result.structuredContent
                        )
                    else:
                        texts = [
                            item.text
                            for item in getattr(result, "content", [])
                            if hasattr(item, "text")
                        ]
                        decoded = self._decode_result("\n".join(texts))
        # Windows Proactor transports finish pipe cleanup on the next loop
        # turns. Give stdio subprocess callbacks time to run before
        # asyncio.run() closes the loop.
        if not config.url:
            await asyncio.sleep(0.1)
        return decoded

    async def list_tools(self, server: str) -> list[dict[str, Any]]:
        return await asyncio.wait_for(
            self._session_call(server, "list"),
            timeout=self.servers[server].timeout_seconds,
        )

    async def call_tool(
        self, server: str, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        config = self.servers[server]
        last_error: Exception | None = None
        for attempt in range(config.retries + 1):
            try:
                return await asyncio.wait_for(
                    self._session_call(
                        server, "call", tool_name, arguments
                    ),
                    timeout=config.timeout_seconds,
                )
            except Exception as error:
                last_error = error
                if attempt < config.retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
        raise RuntimeError(
            f"MCP call failed: {server}.{tool_name}: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def list_tools_sync(self, server: str) -> list[dict[str, Any]]:
        return asyncio.run(self.list_tools(server))

    def call_tool_sync(
        self, server: str, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        return asyncio.run(self.call_tool(server, tool_name, arguments))


def project_mcp_client(root: Path) -> MCPStdioClient:
    root = root.resolve()
    chroma_python = root / ".venv" / "Scripts" / "python.exe"
    graph_python = root / ".venv-raganything" / "Scripts" / "python.exe"
    if not graph_python.is_file():
        graph_python = chroma_python
    pythonpath = str(root / "src")
    hf_home = os.getenv("HF_HOME", "F:\\huggingface")
    graph_env = {
        "PYTHONPATH": pythonpath,
        "HF_HOME": hf_home,
        "HUGGINGFACE_HUB_CACHE": os.getenv(
            "HUGGINGFACE_HUB_CACHE", str(Path(hf_home) / "hub")
        ),
        "HF_HUB_OFFLINE": os.getenv("HF_HUB_OFFLINE", "1"),
        "TRANSFORMERS_OFFLINE": os.getenv("TRANSFORMERS_OFFLINE", "1"),
    }
    return MCPStdioClient(
        {
            "chroma": MCPServerConfig(
                name="chroma",
                command=str(chroma_python),
                args=["-m", "aquabio_mrag.mcp_server"],
                env={"PYTHONPATH": pythonpath},
                timeout_seconds=120,
            ),
            "raganything": MCPServerConfig(
                name="raganything",
                command=str(graph_python),
                args=["-m", "aquabio_raganything.mcp_server"],
                env=graph_env,
                timeout_seconds=float(
                    os.getenv("RAGANYTHING_MCP_TIMEOUT", "35")
                ),
                retries=0,
                url=os.getenv(
                    "RAGANYTHING_MCP_URL",
                    "http://127.0.0.1:8765/mcp",
                ),
            ),
        }
    )
