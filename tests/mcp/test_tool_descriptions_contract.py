from __future__ import annotations

from collections.abc import Callable
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient


def _assert_tool_descriptions(tools) -> None:
    for tool in tools:
        description = (tool.description or "").strip()
        assert description, f"Tool '{tool.name}' must provide a description."
        assert description.startswith("Use this when"), (
            f"Tool '{tool.name}' description should start with 'Use this when...'."
        )


@pytest.mark.asyncio
async def test_read_profile_tools_have_guidance_descriptions(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    async with remote_client_factory(
        transport="streamable-http", profile="read"
    ) as client:
        tools = await client.list_tools_full()
    _assert_tool_descriptions(tools)


@pytest.mark.asyncio
async def test_write_profile_tools_have_guidance_descriptions(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    async with remote_client_factory(
        transport="streamable-http", profile="write"
    ) as client:
        tools = await client.list_tools_full()
    _assert_tool_descriptions(tools)
