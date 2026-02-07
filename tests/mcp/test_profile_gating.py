from __future__ import annotations

from collections.abc import Callable
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, MCPToolError


@pytest.mark.asyncio
async def test_read_profile_excludes_write_tools(
    read_client_factory: Callable[[], AsyncContextManager[MCPClient]],
) -> None:
    async with read_client_factory() as client:
        tools = await client.list_tools()
        assert "build" not in tools
        assert "multi_attempt" not in tools
        assert "run_code" not in tools
        assert "outline" in tools
        assert "search" in tools


@pytest.mark.asyncio
async def test_write_profile_includes_write_tools(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
) -> None:
    async with mcp_client_factory() as client:
        tools = await client.list_tools()
        assert "build" in tools
        assert "multi_attempt" in tools
        assert "run_code" in tools


@pytest.mark.asyncio
async def test_read_profile_rejects_build_call(
    read_client_factory: Callable[[], AsyncContextManager[MCPClient]],
) -> None:
    async with read_client_factory() as client:
        with pytest.raises(MCPToolError):
            await client.call_tool("build", {"clean": False, "output_lines": 5})
