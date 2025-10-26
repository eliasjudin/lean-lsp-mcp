from __future__ import annotations

from types import SimpleNamespace

from conftest import load_from_src

schema = load_from_src("lean_lsp_mcp.schema")
tool_spec = load_from_src("lean_lsp_mcp.tool_spec")
spec_tools = load_from_src("lean_lsp_mcp.server_components.spec_tools")
tool_inputs = load_from_src("lean_lsp_mcp.tool_inputs")


def test_tool_spec_includes_expected_metadata():
    spec = tool_spec.build_tool_spec()
    assert spec["version"] == tool_spec.TOOL_SPEC_VERSION
    assert any(tool["name"] == "lean_goal" for tool in spec["tools"])
    assert "SearchResults" in spec["responses"]

    file_tool = next(tool for tool in spec["tools"] if tool["name"] == "lean_file_contents")
    file_inputs = {item["name"] for item in file_tool["inputs"]}
    assert {"start_line", "line_count"}.issubset(file_inputs)
    assert file_tool["inputModel"] == "LeanFileContentsInput"
    assert (
        file_tool["inputSchema"]["properties"]["file_path"]["description"]
        == "Absolute or project-relative path to a Lean source file."
    )
    annotations = file_tool["annotations"]
    assert annotations["readOnlyHint"] is True
    assert annotations["idempotentHint"] is True
    assert annotations["destructiveHint"] is False
    assert annotations["openWorldHint"] is False
    assert annotations["title"] == "Read Lean File"

    diag_tool = next(tool for tool in spec["tools"] if tool["name"] == "lean_diagnostic_messages")
    diag_inputs = {item["name"] for item in diag_tool["inputs"]}
    assert {"start_line", "line_count"}.issubset(diag_inputs)
    assert diag_tool["inputModel"] == "LeanDiagnosticMessagesInput"
    start_line_schema = diag_tool["inputSchema"]["properties"]["start_line"]
    assert any(
        isinstance(option, dict) and option.get("minimum") == 1
        for option in start_line_schema.get("anyOf", [])
    )
    assert diag_tool["annotations"]["readOnlyHint"] is True
    assert diag_tool["annotations"]["idempotentHint"] is True

    build_tool = next(tool for tool in spec["tools"] if tool["name"] == "lean_build")
    build_annotations = build_tool["annotations"]
    assert build_annotations["readOnlyHint"] is False
    assert build_annotations["idempotentHint"] is True
    assert build_annotations["destructiveHint"] is True
    assert build_annotations["openWorldHint"] is False
    assert build_annotations["title"] == "Rebuild Lean Project"


def test_tool_spec_tool_returns_summary_and_resource():
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=SimpleNamespace()))
    params = tool_inputs.LeanToolSpecInput(response_format=None)

    result = spec_tools.tool_spec(ctx, params)

    assert result["isError"] is False
    assert result["content"][0]["type"] == "text"
    assert "tool specification" in result["content"][0]["text"].lower()

    resource = next(item for item in result["content"] if item["type"] == "resource")
    assert resource["resource"]["mimeType"] == "application/json"
    assert result["structuredContent"]["version"] == tool_spec.TOOL_SPEC_VERSION
