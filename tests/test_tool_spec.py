from __future__ import annotations

from lean_lsp_mcp.tool_spec import build_tool_spec, TOOL_SPEC_VERSION
from lean_lsp_mcp.schema import SCHEMA_VERSION


def test_tool_spec_includes_expected_metadata():
    spec = build_tool_spec()
    assert spec["version"] == TOOL_SPEC_VERSION
    assert spec["schema_version"] == SCHEMA_VERSION
    assert any(tool["name"] == "lean_goal" for tool in spec["tools"])
    assert "SearchResults" in spec["outputs"]

