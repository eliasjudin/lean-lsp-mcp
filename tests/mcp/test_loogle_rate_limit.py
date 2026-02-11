from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import FastMCP

import lean_lsp_mcp.tools_external as tools_external
from lean_lsp_mcp.models import LoogleResult, LoogleResults
from lean_lsp_mcp.tools_external import register_external_tools


class _LoggerStub:
    def warning(self, *_args, **_kwargs) -> None:
        return


class _LocalLoogleStub:
    def __init__(self, *, raise_on_query: bool = False) -> None:
        self.project_path = Path("/tmp/project")
        self.raise_on_query = raise_on_query
        self.query_calls = 0
        self.stop_calls = 0
        self.set_project_path_calls = 0

    def set_project_path(self, project_path: Path) -> bool:
        self.project_path = project_path
        self.set_project_path_calls += 1
        return False

    async def stop(self) -> None:
        self.stop_calls += 1

    async def query(self, query: str, num_results: int) -> list[dict[str, str]]:
        self.query_calls += 1
        if self.raise_on_query:
            raise RuntimeError("local loogle failed")
        return [{"name": query, "type": f"type{num_results}", "module": "Test"}]


class _CtxStub:
    def __init__(self, lifespan_context) -> None:
        self.request_context = SimpleNamespace(lifespan_context=lifespan_context)

    async def report_progress(self, **_kwargs) -> None:
        return


def _register_loogle_tool():
    mcp = FastMCP("test-loogle-rate-limit")
    register_external_tools(
        mcp,
        rg_available=True,
        rg_message="",
        logger=_LoggerStub(),
    )
    return mcp._tool_manager.get_tool("loogle").fn


@pytest.mark.asyncio
async def test_local_loogle_bypasses_remote_rate_limit(monkeypatch: pytest.MonkeyPatch):
    remote_calls = 0

    def _unexpected_remote(_query: str, _num_results: int):
        nonlocal remote_calls
        remote_calls += 1
        return [LoogleResult(name="remote", type="remote", module="remote")]

    monkeypatch.setattr(tools_external, "loogle_remote", _unexpected_remote)

    manager = _LocalLoogleStub()
    app_ctx = SimpleNamespace(
        loogle_local_available=True,
        loogle_manager=manager,
        lean_project_path=Path("/tmp/project"),
        rate_limit={"loogle": deque()},
    )
    ctx = _CtxStub(app_ctx)
    loogle = _register_loogle_tool()

    for _ in range(4):
        result = await loogle(ctx, query="Nat.succ", num_results=1)
        assert isinstance(result, LoogleResults)
        assert result.items[0].name == "Nat.succ"

    assert manager.query_calls == 4
    assert remote_calls == 0


@pytest.mark.asyncio
async def test_remote_loogle_stays_rate_limited(monkeypatch: pytest.MonkeyPatch):
    remote_calls = 0

    def _remote(_query: str, _num_results: int):
        nonlocal remote_calls
        remote_calls += 1
        return [LoogleResult(name="remote", type="remote", module="remote")]

    monkeypatch.setattr(tools_external, "loogle_remote", _remote)

    app_ctx = SimpleNamespace(
        loogle_local_available=False,
        loogle_manager=None,
        lean_project_path=Path("/tmp/project"),
        rate_limit={"loogle": deque()},
    )
    ctx = _CtxStub(app_ctx)
    loogle = _register_loogle_tool()

    for _ in range(3):
        result = await loogle(ctx, query="Nat.succ", num_results=1)
        assert isinstance(result, LoogleResults)

    limited = await loogle(ctx, query="Nat.succ", num_results=1)
    assert isinstance(limited, str)
    assert "Tool limit exceeded" in limited
    assert remote_calls == 3


@pytest.mark.asyncio
async def test_local_failure_falls_back_to_rate_limited_remote(
    monkeypatch: pytest.MonkeyPatch,
):
    remote_calls = 0

    def _remote(_query: str, _num_results: int):
        nonlocal remote_calls
        remote_calls += 1
        return [LoogleResult(name="remote", type="remote", module="remote")]

    monkeypatch.setattr(tools_external, "loogle_remote", _remote)

    manager = _LocalLoogleStub(raise_on_query=True)
    app_ctx = SimpleNamespace(
        loogle_local_available=True,
        loogle_manager=manager,
        lean_project_path=Path("/tmp/project"),
        rate_limit={"loogle": deque()},
    )
    ctx = _CtxStub(app_ctx)
    loogle = _register_loogle_tool()

    for _ in range(3):
        result = await loogle(ctx, query="Nat.succ", num_results=1)
        assert isinstance(result, LoogleResults)

    limited = await loogle(ctx, query="Nat.succ", num_results=1)
    assert isinstance(limited, str)
    assert "Tool limit exceeded" in limited
    assert manager.query_calls == 4
    assert remote_calls == 3
