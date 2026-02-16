from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import FastMCP

import lean_lsp_mcp.tools_external as tools_external
from lean_lsp_mcp.models import LocalSearchResults
from lean_lsp_mcp.tools_external import register_external_tools


class _LoggerStub:
    def warning(self, *_args, **_kwargs) -> None:
        return


class _CtxStub:
    def __init__(self, lifespan_context) -> None:
        self.request_context = SimpleNamespace(lifespan_context=lifespan_context)


def _register_local_search_tool():
    mcp = FastMCP("test-local-search-line")
    register_external_tools(
        mcp,
        rg_available=True,
        rg_message="",
        logger=_LoggerStub(),
    )
    return mcp._tool_manager.get_tool("local_search").fn


@pytest.mark.asyncio
async def test_local_search_includes_line_anchor_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_local_search(*, query: str, limit: int, project_root: Path):
        assert query == "Nat"
        assert limit == 10
        assert project_root == Path("/tmp/project")
        return [
            {"name": "Nat.succ", "kind": "def", "file": "Init/Prelude.lean", "line": 42},
            {"name": "Nat.add", "kind": "theorem", "file": "Mathlib/Data/Nat.lean"},
            {"name": "Nat.mul", "kind": "theorem", "file": "Mathlib/Data/Nat.lean", "line": 0},
        ]

    monkeypatch.setattr(tools_external, "lean_local_search", _fake_local_search)

    app_ctx = SimpleNamespace(workspace_root=Path("/tmp/project"))
    ctx = _CtxStub(app_ctx)
    local_search = _register_local_search_tool()

    result = await local_search(ctx, query="Nat", limit=10)

    assert isinstance(result, LocalSearchResults)
    assert result.items[0].line == 42
    assert result.items[1].line is None
    assert result.items[2].line is None
