from __future__ import annotations

from collections.abc import Callable
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_text


@pytest.mark.asyncio
async def test_rate_limited_errors_propagate(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
) -> None:
    async with mcp_client_factory() as client:
        assert "_test_rate_limited" in await client.list_tools()

        first = await client.call_tool("_test_rate_limited", {})
        assert "ok" in result_text(first)

        second = await client.call_tool("_test_rate_limited", {}, expect_error=True)
        assert second.isError
        assert "Tool limit exceeded" in result_text(second)
