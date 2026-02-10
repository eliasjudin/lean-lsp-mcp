from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest
from mcp.types import TextContent

from lean_lsp_mcp.models import LocalSearchResult, LocalSearchResults
from lean_lsp_mcp.search_fetch import (
    build_canonical_url,
    declaration_text_for_id,
    decode_declaration_id,
    encode_declaration_id,
    search_payload_from_local_results,
)
from lean_lsp_mcp.utils import LeanToolError
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
    assert decoded["line"] is None


def test_search_payload_preserves_line_anchor(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inside = workspace / "Inside.lean"
    inside.write_text("def inside : Nat := 0\n", encoding="utf-8")

    payload = search_payload_from_local_results(
        LocalSearchResults(
            items=[
                LocalSearchResult(name="inside", kind="def", file="Inside.lean", line=1)
            ]
        ),
        workspace,
    )

    assert len(payload.results) == 1
    decoded = decode_declaration_id(payload.results[0].id)
    assert decoded["line"] == 1


def test_fetch_line_anchored_id_does_not_require_lsp_client(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inside = workspace / "Inside.lean"
    inside.write_text(
        "def inside : Nat := 0\n\n"
        "theorem t : True := by\n"
        "  trivial\n\n"
        "def after : Nat := 1\n",
        encoding="utf-8",
    )

    identifier = encode_declaration_id(path="Inside.lean", symbol="t", line=3)
    payload = declaration_text_for_id(
        workspace_root=workspace,
        client=None,
        identifier=identifier,
    )

    assert payload.id == identifier
    assert payload.title == "t"
    assert payload.text == "theorem t : True := by\n  trivial"


def test_fetch_line_anchored_id_keeps_nested_where_declarations(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inside = workspace / "Inside.lean"
    inside.write_text(
        "theorem outer : True := by\n"
        "  trivial\n"
        "where\n"
        "  def helper : Nat := 1\n"
        "  theorem helper_ok : True := by\n"
        "    trivial\n\n"
        "def after : Nat := 2\n",
        encoding="utf-8",
    )

    identifier = encode_declaration_id(path="Inside.lean", symbol="outer", line=1)
    payload = declaration_text_for_id(
        workspace_root=workspace,
        client=None,
        identifier=identifier,
    )

    assert payload.text.startswith("theorem outer : True := by")
    assert "def helper : Nat := 1" in payload.text
    assert "theorem helper_ok : True := by" in payload.text
    assert "def after : Nat := 2" not in payload.text


def test_build_canonical_url_rejects_non_http_scheme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LEAN_PUBLIC_BASE_URL", "javascript:alert(1)")
    with pytest.raises(LeanToolError):
        build_canonical_url("abc")


def test_build_canonical_url_rejects_non_local_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LEAN_PUBLIC_BASE_URL", "http://example.com")
    with pytest.raises(LeanToolError):
        build_canonical_url("abc")


def test_build_canonical_url_allows_https(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEAN_PUBLIC_BASE_URL", "https://docs.example.com")
    url = build_canonical_url("abc")
    assert url == "https://docs.example.com/decl/abc"


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
