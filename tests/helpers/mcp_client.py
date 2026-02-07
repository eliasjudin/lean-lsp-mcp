from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

try:
    from mcp.client.streamable_http import streamable_http_client
except ImportError:  # mcp<1.26 compatibility
    streamable_http_client = None  # type: ignore[assignment]
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

    def __init__(self, session: ClientSession, *, endpoint: str | None = None) -> None:
        self._session = session
        self.endpoint = endpoint
        if endpoint and endpoint.endswith("/mcp"):
            self.base_url = endpoint[: -len("/mcp")]
        elif endpoint and endpoint.endswith("/sse"):
            self.base_url = endpoint[: -len("/sse")]
        else:
            self.base_url = endpoint

    async def list_tools(self) -> list[str]:
        result = await self._session.list_tools()
        return [tool.name for tool in result.tools]

    async def list_tools_full(self):
        result = await self._session.list_tools()
        return list(result.tools)

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
    if streamable_http_client is None:
        async with streamablehttp_client(url, headers=headers) as (
            read_stream,
            write_stream,
            _,
        ):
            session = ClientSession(read_stream, write_stream)
            async with session:
                await session.initialize()
                yield MCPClient(session, endpoint=url)
        return

    http_client: httpx.AsyncClient | None = None
    if headers:
        http_client = httpx.AsyncClient(headers=headers)
    try:
        async with streamable_http_client(url, http_client=http_client) as (
            read_stream,
            write_stream,
            _,
        ):
            session = ClientSession(read_stream, write_stream)
            async with session:
                await session.initialize()
                yield MCPClient(session, endpoint=url)
    finally:
        if http_client is not None:
            await http_client.aclose()


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
            yield MCPClient(session, endpoint=url)


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
