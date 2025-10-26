from __future__ import annotations

from types import SimpleNamespace

import pytest
from conftest import load_from_src

schema = load_from_src("lean_lsp_mcp.schema")
server = load_from_src("lean_lsp_mcp.server")
formatter = load_from_src("lean_lsp_mcp.response_formatter")


def test_mcp_result_builds_structured_envelope():
    result = schema.mcp_result(
        content=[{"type": "text", "text": "hello"}],
        structured={"value": 1},
    )

    assert result["content"] == [{"type": "text", "text": "hello"}]
    assert result["structuredContent"]["value"] == 1
    assert result["isError"] is False


def test_mcp_result_requires_content():
    with pytest.raises(ValueError):
        schema.mcp_result(content=[])


def test_mcp_result_requires_mapping_items():
    with pytest.raises(TypeError):
        schema.mcp_result(content=[{"type": "text", "text": "hello"}, "oops"])


def test_mcp_result_requires_mapping_structured_payload():
    with pytest.raises(TypeError):
        schema.mcp_result(
            content=[{"type": "text", "text": "hello"}],
            structured=[("key", "value")],
        )


def test_mcp_result_requires_serializable_structured_payload():
    with pytest.raises(TypeError):
        schema.mcp_result(
            content=[{"type": "text", "text": "hello"}],
            structured={"value": set()},
        )


def test_mcp_result_accepts_generator_and_sets_error_flag():
    def content_iter():
        yield {"type": "text", "text": "edge"}

    result = schema.mcp_result(content=content_iter(), is_error=True)

    assert result["content"] == [{"type": "text", "text": "edge"}]
    assert result["isError"] is True


def test_mcp_result_structured_payload_is_copied():
    structured = {"value": 3}

    result = schema.mcp_result(
        content=[{"type": "text", "text": "data"}],
        structured=structured,
    )

    structured["value"] = 4

    assert result["structuredContent"]["value"] == 3


def test_success_result_builds_payload_without_meta():
    result = server.success_result(
        summary="done",
        structured={"value": 42},
        start_time=0.0,
        ctx=None,
    )

    # Markdown summary first by default
    first = result["content"][0]
    assert first["type"] == "text"
    assert first["text"].splitlines()[0] == "**Summary:** done"
    assert result["structuredContent"]["value"] == 42
    assert result["isError"] is False
    assert "_meta" not in result["structuredContent"]


def test_error_result_sets_code_and_category_without_meta():
    result = server.error_result(
        message="boom",
        code="sample",
        category="demo",
        details={"info": "extra"},
        start_time=0.0,
        ctx=None,
    )

    assert result["isError"] is True
    # Markdown error summary emitted by default
    assert result["content"][0]["type"] == "text"
    error_lines = result["content"][0]["text"].splitlines()
    assert error_lines[0] == "**Summary:** Error: boom"
    assert any(line == "- Code: `sample`" for line in error_lines[1:])
    assert any(line == "- Category: demo" for line in error_lines[1:])
    assert any(line.startswith("- Hint: ") for line in error_lines[1:])
    structured = result["structuredContent"]
    assert structured["message"] == "boom"
    assert structured["code"] == "sample"
    assert structured["category"] == "demo"
    assert structured["details"] == {"info": "extra"}
    assert structured["hints"]
    assert "_meta" not in structured


def test_success_result_omits_duplicate_summary_block():
    result = server.success_result(
        summary="done",
        structured=None,
        start_time=0.0,
        ctx=None,
        content=[
            {"type": "text", "text": "done"},
            {"type": "text", "text": "details"},
        ],
    )

    texts = [item["text"] for item in result["content"] if item["type"] == "text"]

    assert texts[0].startswith("**Summary:** done")
    assert "done" not in texts[1:]
    assert "details" in texts[1:]


def test_success_result_truncates_and_emits_hint():
    oversized = "x" * (formatter.CHARACTER_LIMIT + 200)
    result = server.success_result(
        summary="done",
        structured=None,
        start_time=0.0,
        ctx=None,
        content=[{"type": "text", "text": oversized}],
    )

    # Expect summary preserved and truncation hint appended.
    summary_line = result["content"][0]["text"].splitlines()[0]
    assert summary_line == "**Summary:** done"
    truncation_note = result["content"][-1]["text"]
    assert truncation_note.startswith("_Note: Output truncated")

    total_text = "".join(
        content_item["text"]
        for content_item in result["content"]
        if content_item["type"] == "text"
    )
    assert len(total_text) <= formatter.CHARACTER_LIMIT

    structured = result.get("structuredContent")
    assert structured is not None
    meta = structured["_meta"]
    assert meta["truncated"] is True
    assert meta["character_limit"] == formatter.CHARACTER_LIMIT
    assert meta["truncation_hint"].startswith("Output truncated to")


def test_success_result_json_format_embeds_summary_meta():
    result = server.success_result(
        summary="done",
        structured={"value": 7},
        start_time=0.0,
        ctx=None,
        response_format="json",
    )

    first = result["content"][0]
    assert first["type"] == "resource"
    assert first["resource"]["mimeType"] == "application/json"
    assert result["structuredContent"]["value"] == 7
    meta = result["structuredContent"].get("_meta")
    assert meta["summary"] == "done"
    assert result["content"][0]["resource"]["text"].startswith("{")


def test_error_result_json_format_includes_summary_meta():
    result = server.error_result(
        message="boom",
        code="sample",
        category="demo",
        details={"info": "extra"},
        start_time=0.0,
        ctx=None,
        response_format="json",
    )

    assert result["content"][0]["type"] == "resource"
    meta = result["structuredContent"].get("_meta")
    assert meta["summary"] == "Error: boom"


def test_error_result_deduplicates_hints():
    lifespan = SimpleNamespace(lean_project_path="/tmp/project", rate_limit={})
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=lifespan))
    derived_hint = server._derive_error_hints(
        code=server.ERROR_CLIENT_NOT_READY,
        category=None,
        details={},
        ctx=ctx,
    )[0]

    result = server.error_result(
        message="client missing",
        code=server.ERROR_CLIENT_NOT_READY,
        category=None,
        details=None,
        hints=[derived_hint, derived_hint],
        start_time=0.0,
        ctx=ctx,
    )

    hints = result["structuredContent"]["hints"]

    assert hints.count(derived_hint) == 1
