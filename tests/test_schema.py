from __future__ import annotations

import os
from typing import Any, Dict

from lean_lsp_mcp.schema import make_response, SCHEMA_VERSION, RESPONSE_FORMAT_ENV


def test_make_response_structured(monkeypatch):
    monkeypatch.delenv(RESPONSE_FORMAT_ENV, raising=False)
    payload = {"value": 1}
    response = make_response("ok", data=payload)
    assert response["status"] == "ok"
    assert response["data"] == payload
    assert response["meta"]["schema_version"] == SCHEMA_VERSION


def test_make_response_legacy(monkeypatch):
    monkeypatch.setenv(RESPONSE_FORMAT_ENV, "legacy")
    payload = {"value": 2}

    def formatter(envelope: Dict[str, Any]) -> str:
        return f"legacy:{envelope['data']['value']}"

    response = make_response("ok", data=payload, legacy_formatter=formatter)
    assert response == "legacy:2"


