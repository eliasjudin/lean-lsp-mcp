from __future__ import annotations

import time
import types

import pytest

from conftest import load_from_src

schema = load_from_src("lean_lsp_mcp.schema")
server = load_from_src("lean_lsp_mcp.server")


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


def test_success_result_builds_payload():
    started = time.perf_counter()
    result = server.success_result(
        summary="done",
        structured={"value": 42},
        start_time=started,
        ctx=None,
    )

    # JSON resource first, then human summary
    first = result["content"][0]
    assert first["type"] == "resource"
    assert first["resource"]["mimeType"] == "application/json"
    assert result["content"][1]["text"] == "done"
    assert result["structuredContent"]["value"] == 42
    assert result["isError"] is False
    meta = result["_meta"]
    assert meta["duration_ms"] >= 0
    assert "request_id" not in meta


def test_error_result_sets_code_and_category():
    started = time.perf_counter()
    result = server.error_result(
        message="boom",
        code="sample",
        category="demo",
        details={"info": "extra"},
        start_time=started,
        ctx=None,
    )

    assert result["isError"] is True
    # JSON resource first, then text message
    assert result["content"][0]["type"] == "resource"
    assert result["content"][0]["resource"]["mimeType"] == "application/json"
    assert result["content"][1]["text"] == "boom"
    structured = result["structuredContent"]
    assert structured["message"] == "boom"
    assert structured["code"] == "sample"
    assert structured["category"] == "demo"
    assert structured["details"] == {"info": "extra"}
    meta = result["_meta"]
    assert meta["duration_ms"] >= 0
    assert "request_id" not in meta


def test_meta_uses_request_context_request_id():
    ctx = types.SimpleNamespace(
        request_context=types.SimpleNamespace(request_id="req-123")
    )
    started = time.perf_counter()
    result = server.success_result(
        summary="ok",
        structured=None,
        start_time=started,
        ctx=ctx,
        content=[{"type": "text", "text": "ok"}],
    )

    meta = result["_meta"]
    assert meta["request_id"] == "req-123"
    assert meta["duration_ms"] >= 0
