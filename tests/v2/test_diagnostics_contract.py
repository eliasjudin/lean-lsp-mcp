from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, result_json


@pytest.mark.asyncio
async def test_diagnostics_success_true_for_clean_file(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
) -> None:
    async with mcp_client_factory() as client:
        result = await client.call_tool("diagnostics", {"path": "McpTestProject.lean"})
        payload = result_json(result)
        assert payload["success"] is True


@pytest.mark.asyncio
async def test_diagnostics_success_false_for_error_file(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    broken_name = "BrokenDiagnostics.lean"
    broken_path = test_project_path / broken_name
    broken_path.write_text("def broken :=\n", encoding="utf-8")

    try:
        async with mcp_client_factory() as client:
            result = await client.call_tool("diagnostics", {"path": broken_name})
            payload = result_json(result)

        assert payload["success"] is False
        assert any(item.get("severity") == "error" for item in payload["items"])
    finally:
        broken_path.unlink(missing_ok=True)
