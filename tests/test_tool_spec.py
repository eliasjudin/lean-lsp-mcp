from __future__ import annotations

from conftest import load_module

schema = load_module("lean_lsp_mcp.schema")
tool_spec = load_module("lean_lsp_mcp.tool_spec")


def test_tool_spec_includes_expected_metadata():
    spec = tool_spec.build_tool_spec()
    assert spec["version"] == tool_spec.TOOL_SPEC_VERSION
    assert spec["schema_version"] == schema.SCHEMA_VERSION
    assert any(tool["name"] == "lean_goal" for tool in spec["tools"])
    assert "SearchResults" in spec["outputs"]
