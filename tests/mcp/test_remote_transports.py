from __future__ import annotations

import json
from collections.abc import Callable
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_text


@pytest.mark.asyncio
async def test_streamable_http_roundtrip(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
) -> None:
    async with mcp_client_factory() as client:
        tools = await client.list_tools()
        assert "outline" in tools
        result = await client.call_tool("search", {"query": "sampleTheorem"})
        payload = json.loads(result_text(result))
        assert payload["results"]


@pytest.mark.asyncio
async def test_sse_roundtrip(
    sse_client_factory: Callable[[], AsyncContextManager[MCPClient]],
) -> None:
    async with sse_client_factory() as client:
        tools = await client.list_tools()
        assert "outline" in tools
        result = await client.call_tool("search", {"query": "sampleTheorem"})
        payload = json.loads(result_text(result))
        assert payload["results"]
