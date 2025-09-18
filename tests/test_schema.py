from __future__ import annotations

import importlib.util
from pathlib import Path

from conftest import ensure_mcp_stub

ensure_mcp_stub()

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "lean_lsp_mcp" / "schema.py"
spec = importlib.util.spec_from_file_location("lean_lsp_mcp.schema", MODULE_PATH)
schema = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(schema)


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
