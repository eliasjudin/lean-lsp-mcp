from __future__ import annotations

import importlib.util
from pathlib import Path

from conftest import ensure_mcp_stub

ensure_mcp_stub()

SRC = Path(__file__).resolve().parents[1] / "src" / "lean_lsp_mcp"

schema_spec = importlib.util.spec_from_file_location("lean_lsp_mcp.schema", SRC / "schema.py")
schema = importlib.util.module_from_spec(schema_spec)
assert schema_spec.loader is not None
schema_spec.loader.exec_module(schema)

tool_spec_spec = importlib.util.spec_from_file_location("lean_lsp_mcp.tool_spec", SRC / "tool_spec.py")
tool_spec = importlib.util.module_from_spec(tool_spec_spec)
assert tool_spec_spec.loader is not None
tool_spec_spec.loader.exec_module(tool_spec)


def test_tool_spec_includes_expected_metadata():
    spec = tool_spec.build_tool_spec()
    assert spec["version"] == tool_spec.TOOL_SPEC_VERSION
    assert spec["schema_version"] == schema.SCHEMA_VERSION
    assert any(tool["name"] == "lean_goal" for tool in spec["tools"])
    assert "SearchResults" in spec["responses"]
