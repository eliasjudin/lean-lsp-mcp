from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_text


@pytest.mark.asyncio
async def test_file_based_tools_use_relative_path(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    path = "EditorTools.lean"

    async with mcp_client_factory() as client:
        outline = await client.call_tool("outline", {"path": path})
        assert "imports" in result_text(outline)

        diags = await client.call_tool("diagnostics", {"path": path})
        assert "items" in result_text(diags)

        goal = await client.call_tool("goal", {"path": path, "line": 4})
        assert "goals" in result_text(goal)

        hover = await client.call_tool("hover", {"path": path, "line": 3, "column": 9})
        assert "sampleTheorem" in result_text(hover)


@pytest.mark.asyncio
async def test_path_escape_rejected(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
) -> None:
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "outline", {"path": "../outside.lean"}, expect_error=True
        )
        assert result.isError


@pytest.mark.asyncio
async def test_write_tools_operate_in_write_profile(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
) -> None:
    async with mcp_client_factory() as client:
        run = await client.call_tool(
            "run_code",
            {
                "code": "#eval Nat.succ 0\n",
            },
        )
        assert run.structuredContent is not None
        assert "diagnostics" in run.structuredContent

        multi = await client.call_tool(
            "multi_attempt",
            {
                "path": "MiscTools.lean",
                "line": 5,
                "snippets": ["rfl", "simp"],
            },
        )
        text = result_text(multi)
        assert "rfl" in text or "simp" in text
