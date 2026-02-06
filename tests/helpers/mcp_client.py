from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import (
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    ResourceLink,
    TextContent,
)


class MCPClientError(RuntimeError):
    """Base exception raised for MCP helper failures."""


@dataclass(slots=True)
class MCPToolError(MCPClientError):
    """Exception raised when a tool returns an error payload."""

    tool_name: str
    result: CallToolResult

    def __str__(self) -> str:  # pragma: no cover - trivial string repr
        return f"Tool '{self.tool_name}' failed: {result_text(self.result)}"


class MCPClient:
    """Lightweight helper around :class:`mcp.client.session.ClientSession`."""

    def __init__(self, session: ClientSession) -> None:
        self._session = session

    async def list_tools(self) -> list[str]:
        result = await self._session.list_tools()
        return [tool.name for tool in result.tools]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        expect_error: bool = False,
    ) -> CallToolResult:
        result = await self._session.call_tool(name, arguments or {})
        if result.isError and not expect_error:
            raise MCPToolError(name, result)
        return result


@asynccontextmanager
async def connect_streamable_http_client(
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> AsyncIterator[MCPClient]:
    async with streamablehttp_client(url, headers=headers) as (
        read_stream,
        write_stream,
        _,
    ):
        session = ClientSession(read_stream, write_stream)
        async with session:
            await session.initialize()
            yield MCPClient(session)


@asynccontextmanager
async def connect_sse_client(
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> AsyncIterator[MCPClient]:
    async with sse_client(url, headers=headers) as (read_stream, write_stream):
        session = ClientSession(read_stream, write_stream)
        async with session:
            await session.initialize()
            yield MCPClient(session)


def result_text(result: CallToolResult) -> str:
    segments: list[str] = []
    for block in result.content:
        segments.extend(_text_from_block(block))
    return "\n".join(segment for segment in segments if segment)


def result_json(result: CallToolResult) -> dict[str, Any]:
    return json.loads(result_text(result))


def _text_from_block(block: ContentBlock) -> Iterable[str]:
    if isinstance(block, TextContent):
        yield block.text
    elif isinstance(block, ResourceLink):
        yield block.uri
    elif isinstance(block, EmbeddedResource):
        resource = block.resource
        if hasattr(resource, "text"):
            yield resource.text  # type: ignore[no-any-return]
    else:
        yield ""
