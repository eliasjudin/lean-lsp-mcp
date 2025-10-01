from __future__ import annotations

from conftest import load_from_src

schema = load_from_src("lean_lsp_mcp.schema")
tool_spec = load_from_src("lean_lsp_mcp.tool_spec")


def test_tool_spec_includes_expected_metadata():
    spec = tool_spec.build_tool_spec()
    assert spec["version"] == tool_spec.TOOL_SPEC_VERSION
    assert any(tool["name"] == "lean_goal" for tool in spec["tools"])
    assert "SearchResults" in spec["responses"]

    file_tool = next(tool for tool in spec["tools"] if tool["name"] == "lean_file_contents")
    file_inputs = {item["name"] for item in file_tool["inputs"]}
    assert {"start_line", "line_count"}.issubset(file_inputs)

    diag_tool = next(tool for tool in spec["tools"] if tool["name"] == "lean_diagnostic_messages")
    diag_inputs = {item["name"] for item in diag_tool["inputs"]}
    assert {"start_line", "line_count"}.issubset(diag_inputs)
