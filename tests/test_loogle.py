from __future__ import annotations

import json
import urllib.request
from types import SimpleNamespace

import pytest

from conftest import load_from_src


class DummyResponse:
    """Minimal response stub for urllib.request.urlopen."""

    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, *_args) -> None:  # pragma: no cover - no cleanup needed
        return None


def make_ctx() -> SimpleNamespace:
    lifespan = SimpleNamespace(
        rate_limit={
            "loogle": [],
            "leansearch": [],
        }
    )
    return SimpleNamespace(request_context=SimpleNamespace(lifespan_context=lifespan))


@pytest.fixture()
def server_module():
    return load_from_src("lean_lsp_mcp.server")


def test_loogle_handles_missing_doc(monkeypatch, server_module):
    payload = {
        "hits": [
            {
                "name": "Real.sin",
                "type": "ℝ → ℝ",
                "module": "Mathlib.Data.Complex.Trigonometric",
                # intentionally omit "doc" to ensure code handles absence
            }
        ]
    }

    def fake_urlopen(_req, timeout=20):  # pragma: no cover - tested via tool call
        assert timeout == 20
        return DummyResponse(payload)

    monkeypatch.setattr(server_module.urllib.request, "urlopen", fake_urlopen)

    ctx = make_ctx()
    response = server_module.loogle(ctx=ctx, query="Real.sin", num_results=3)

    assert response["isError"] is False
    structured = response["structuredContent"]
    assert structured["query"] == "Real.sin"
    assert structured["names"] == ["Real.sin"]


def test_loogle_truncates_results_and_strips_doc(monkeypatch, server_module):
    payload = {
        "hits": [
            {
                "name": "foo",
                "doc": "should be removed",
            },
            {
                "name": "bar",
                "doc": None,
            },
        ]
    }

    def fake_urlopen(_req, timeout=20):  # pragma: no cover - tested via tool call
        return DummyResponse(payload)

    monkeypatch.setattr(server_module.urllib.request, "urlopen", fake_urlopen)

    ctx = make_ctx()
    response = server_module.loogle(ctx=ctx, query="demo", num_results=1)

    assert response["isError"] is False
    structured = response["structuredContent"]
    assert structured["query"] == "demo"
    assert structured["names"] == ["foo"]


def test_loogle_defaults_to_compact(monkeypatch, server_module):
    payload = {
        "hits": [
            {
                "name": "foo",
                "doc": "ignored",
            }
        ]
    }

    def fake_urlopen(_req, timeout=20):  # pragma: no cover - tested via tool call
        return DummyResponse(payload)

    monkeypatch.setattr(server_module.urllib.request, "urlopen", fake_urlopen)

    ctx = make_ctx()
    response = server_module.loogle(ctx=ctx, query="demo", num_results=1)

    assert response["isError"] is False
    structured = response["structuredContent"]
    assert structured["query"] == "demo"
    assert structured["names"] == ["foo"]
    assert "results" not in structured


def test_loogle_returns_error_when_no_results(monkeypatch, server_module):
    payload = {"hits": []}

    def fake_urlopen(_req, timeout=20):  # pragma: no cover - tested via tool call
        return DummyResponse(payload)

    monkeypatch.setattr(server_module.urllib.request, "urlopen", fake_urlopen)

    ctx = make_ctx()
    response = server_module.loogle(ctx=ctx, query="missing", num_results=2)

    assert response["isError"] is True
    # JSON resource first, then message
    assert response["content"][0]["type"] == "resource"
    assert response["content"][0]["resource"]["mimeType"] == "application/json"
    assert response["content"][1]["text"] == "No results found."
    structured = response["structuredContent"]
    assert structured["category"] == "lean_loogle"
    assert structured["code"] == server_module.ERROR_UNKNOWN
    assert structured["details"]["query"] == "missing"


def test_leansearch_handles_missing_docstring(monkeypatch, server_module):
    payload = [
        [
            {
                "result": {
                    "name": ["Foo", "bar"],
                    "module_name": ["Mathlib", "Demo"],
                }
            }
        ]
    ]

    def fake_urlopen(_req, timeout=20):  # pragma: no cover - exercised via tool call
        return DummyResponse(payload)

    monkeypatch.setattr(server_module.urllib.request, "urlopen", fake_urlopen)

    ctx = make_ctx()
    response = server_module.leansearch(ctx=ctx, query="Foo", num_results=5)

    assert response["isError"] is False
    structured = response["structuredContent"]
    assert structured["query"] == "Foo"
    assert structured["names"] == ["Foo.bar"]


def test_leansearch_defaults_to_compact(monkeypatch, server_module):
    payload = [
        [
            {
                "result": {
                    "name": ["Foo", "bar"],
                    "module_name": ["Mathlib", "Demo"],
                }
            }
        ]
    ]

    def fake_urlopen(_req, timeout=20):  # pragma: no cover - exercised via tool call
        return DummyResponse(payload)

    monkeypatch.setattr(server_module.urllib.request, "urlopen", fake_urlopen)

    ctx = make_ctx()
    response = server_module.leansearch(ctx=ctx, query="Foo", num_results=5)

    assert response["isError"] is False
    structured = response["structuredContent"]
    assert structured["query"] == "Foo"
    assert structured["names"] == ["Foo.bar"]
    assert "results" not in structured


def test_leansearch_returns_error_when_empty(monkeypatch, server_module):
    payload = [[]]

    def fake_urlopen(_req, timeout=20):  # pragma: no cover - exercised via tool call
        return DummyResponse(payload)

    monkeypatch.setattr(server_module.urllib.request, "urlopen", fake_urlopen)

    ctx = make_ctx()
    response = server_module.leansearch(ctx=ctx, query="nothing", num_results=2)

    assert response["isError"] is True
    # JSON resource first, then message
    content = response["content"][1]
    assert content["text"] == "No results found."
    structured = response["structuredContent"]
    assert structured["category"] == "lean_leansearch"
    assert structured["code"] == server_module.ERROR_UNKNOWN
    assert structured["details"]["query"] == "nothing"
