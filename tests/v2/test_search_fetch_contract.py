from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest
from mcp.types import TextContent

from lean_lsp_mcp.models import LocalSearchResult, LocalSearchResults
from lean_lsp_mcp.v2.search_fetch import (
    decode_declaration_id,
    search_payload_from_local_results,
)
from tests.helpers.mcp_client import MCPClient, result_text


def test_search_payload_filters_unfetchable_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inside = workspace / "Inside.lean"
    inside.write_text("def inside : Nat := 0\n", encoding="utf-8")

    outside = tmp_path / "Outside.lean"
    outside.write_text("def outside : Nat := 0\n", encoding="utf-8")
    symlink = workspace / "Link.lean"
    try:
        symlink.symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks not supported on this platform")

    payload = search_payload_from_local_results(
        LocalSearchResults(
            items=[
                LocalSearchResult(name="inside", kind="def", file="Inside.lean"),
                LocalSearchResult(name="outside", kind="def", file=str(outside)),
                LocalSearchResult(name="link", kind="def", file="Link.lean"),
            ]
        ),
        workspace,
    )

    assert len(payload.results) == 1
    assert payload.results[0].title == "inside"
    decoded = decode_declaration_id(payload.results[0].id)
    assert Path(decoded["path"]) == Path("Inside.lean")


@pytest.mark.asyncio
async def test_search_fetch_payload_contract(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
) -> None:
    async with mcp_client_factory() as client:
        search_result = await client.call_tool("search", {"query": "sampleTheorem"})
        assert len(search_result.content) == 1
        assert isinstance(search_result.content[0], TextContent)
        payload = json.loads(result_text(search_result))
        assert "results" in payload
        assert isinstance(payload["results"], list)
        assert payload["results"]
        first = payload["results"][0]
        assert {"id", "title", "url"}.issubset(first.keys())

        fetch_result = await client.call_tool("fetch", {"id": first["id"]})
        assert len(fetch_result.content) == 1
        assert isinstance(fetch_result.content[0], TextContent)
        fetch_payload = json.loads(result_text(fetch_result))
        assert {"id", "title", "text", "url"}.issubset(fetch_payload.keys())
        assert fetch_payload["id"] == first["id"]
        assert isinstance(fetch_payload["text"], str)


@pytest.mark.asyncio
async def test_search_schema_rejects_extra_fields(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
) -> None:
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "search",
            {"query": "sampleTheorem", "extra": "x"},
            expect_error=True,
        )
        assert result.isError


@pytest.mark.asyncio
async def test_fetch_schema_rejects_extra_fields(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
) -> None:
    async with mcp_client_factory() as client:
        result = await client.call_tool(
            "fetch",
            {"id": "abc", "extra": "x"},
            expect_error=True,
        )
        assert result.isError
