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
    params = server_module.LoogleSearchInput(query="Real.sin", num_results=3)
    response = server_module.loogle(ctx, params)

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
    params = server_module.LoogleSearchInput(query="demo", num_results=1)
    response = server_module.loogle(ctx, params)

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
    params = server_module.LoogleSearchInput(query="demo", num_results=1)
    response = server_module.loogle(ctx, params)

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
    params = server_module.LoogleSearchInput(query="missing", num_results=2)
    response = server_module.loogle(ctx, params)

    assert response["isError"] is True
    # Markdown error summary is first by default
    summary_block = response["content"][0]
    assert summary_block["type"] == "text"
    error_lines = summary_block["text"].splitlines()
    assert error_lines[0] == "**Summary:** Error: No results found."
    assert "- Code: `unknown`" in error_lines[1]
    assert "- Category: lean_loogle" in error_lines[2]
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
    params = server_module.LeanSearchInput(query="Foo", num_results=5)
    response = server_module.leansearch(ctx, params)

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
    params = server_module.LeanSearchInput(query="Foo", num_results=5)
    response = server_module.leansearch(ctx, params)

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
    params = server_module.LeanSearchInput(query="nothing", num_results=2)
    response = server_module.leansearch(ctx, params)

    assert response["isError"] is True
    summary_block = response["content"][0]
    assert summary_block["type"] == "text"
    error_lines = summary_block["text"].splitlines()
    assert error_lines[0] == "**Summary:** Error: No results found."
    assert "- Code: `unknown`" in error_lines[1]
    assert "- Category: lean_leansearch" in error_lines[2]
    structured = response["structuredContent"]
    assert structured["category"] == "lean_leansearch"
    assert structured["code"] == server_module.ERROR_UNKNOWN
    assert structured["details"]["query"] == "nothing"
