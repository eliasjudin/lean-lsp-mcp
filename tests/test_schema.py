from __future__ import annotations

from conftest import load_from_src

schema = load_from_src("lean_lsp_mcp.schema")
server = load_from_src("lean_lsp_mcp.server")


def test_make_response_structured(monkeypatch):
    monkeypatch.delenv(schema.RESPONSE_FORMAT_ENV, raising=False)
    payload = {"value": 1}
    response = schema.make_response("ok", data=payload)
    assert response["status"] == "ok"
    assert response["data"] == payload
    assert response["meta"]["schema_version"] == schema.SCHEMA_VERSION


def test_make_response_legacy(monkeypatch):
    monkeypatch.setenv(schema.RESPONSE_FORMAT_ENV, "legacy")
    payload = {"value": 2}

    def formatter(envelope):
        return f"legacy:{envelope['data']['value']}"

    response = schema.make_response("ok", data=payload, legacy_formatter=formatter)
    assert response == "legacy:2"


def test_error_response_sets_code():
    response = server.error_response("boom", code="sample")
    assert response["status"] == "error"
    assert response["data"]["message"] == "boom"
    assert response["meta"]["error"]["code"] == "sample"
