from __future__ import annotations

from conftest import load_from_src

server = load_from_src("lean_lsp_mcp.server")


def test_compact_pos_returns_one_based_indices() -> None:
    pos = server._compact_pos(line=3, column=5)
    assert pos == {"l": 3, "c": 5}


def test_compact_pos_clamps_to_minimum_one() -> None:
    pos = server._compact_pos(line=0, column=0)
    assert pos == {"l": 1, "c": 1}


def test_compact_pos_omits_missing_dimension() -> None:
    pos = server._compact_pos(line=2)
    assert pos == {"l": 2}
