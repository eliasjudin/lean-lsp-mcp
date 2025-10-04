from __future__ import annotations

from types import SimpleNamespace

import pytest

from conftest import load_from_src

schema = load_from_src("lean_lsp_mcp.schema")
server = load_from_src("lean_lsp_mcp.server")


def test_mcp_result_builds_structured_envelope():
    result = schema.mcp_result(
        content=[{"type": "text", "text": "hello"}],
        structured={"value": 1},
        meta={"duration_ms": 5},
    )

    assert result["content"] == [{"type": "text", "text": "hello"}]
    assert result["structuredContent"]["value"] == 1
    assert result["isError"] is False
    assert result["_meta"]["duration_ms"] == 5


def test_mcp_result_requires_content():
    with pytest.raises(ValueError):
        schema.mcp_result(content=[])


def test_success_result_includes_metadata(monkeypatch):
    fake_uuid = SimpleNamespace(hex="abc123")
    monkeypatch.setattr(server.uuid, "uuid4", lambda: fake_uuid)
    monkeypatch.setattr(server.time, "perf_counter", lambda: 101.5)

    result = server.success_result(
        summary="done",
        structured={"value": 42},
        start_time=100.0,
        meta_extra={"tool": "demo"},
    )

    # JSON resource first, then human summary
    first = result["content"][0]
    assert first["type"] == "resource"
    assert first["resource"]["mimeType"] == "application/json"
    assert result["content"][1]["text"] == "done"
    assert result["structuredContent"]["value"] == 42
    assert result["isError"] is False
    meta = result["_meta"]
    assert meta["duration_ms"] == 1500
    assert meta["request_id"] == "abc123"
    assert meta["tool"] == "demo"


def test_error_result_sets_code_and_category(monkeypatch):
    fake_uuid = SimpleNamespace(hex="err123")
    monkeypatch.setattr(server.uuid, "uuid4", lambda: fake_uuid)
    monkeypatch.setattr(server.time, "perf_counter", lambda: 5.01)

    result = server.error_result(
        message="boom",
        code="sample",
        category="demo",
        details={"info": "extra"},
        start_time=5.0,
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
    assert meta["duration_ms"] == 10
    assert meta["request_id"] == "err123"
