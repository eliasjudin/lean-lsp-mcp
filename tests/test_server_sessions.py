from __future__ import annotations

from threading import Lock
from types import SimpleNamespace

import pytest

from conftest import load_from_src

server = load_from_src("lean_lsp_mcp.server")


def _make_ctx(
    *,
    client=None,
    client_lock=None,
    lean_project_path=None,
    project_cache=None,
    file_content_hashes=None,
):
    lifespan = SimpleNamespace(
        lean_project_path=lean_project_path,
        client=client,
        client_lock=client_lock,
        project_cache=project_cache if project_cache is not None else {},
        file_content_hashes=file_content_hashes if file_content_hashes is not None else {},
        rate_limit={},
    )
    request_context = SimpleNamespace(lifespan_context=lifespan)
    return SimpleNamespace(request_context=request_context), lifespan


def test_open_file_session_yields_session(monkeypatch):
    file_path = "/project/Main.lean"
    identity = {"relative_path": "Main.lean", "uri": "file:///project/Main.lean"}

    captured_calls = {}

    def fake_setup(ctx, path):
        captured_calls["ctx"] = ctx
        captured_calls["path"] = path
        return "Main.lean"

    monkeypatch.setattr(server, "setup_client_for_file", fake_setup)
    monkeypatch.setattr(server, "_identity_for_rel_path", lambda ctx, rel: identity)

    client = SimpleNamespace()
    lock = Lock()
    ctx, lifespan = _make_ctx(client=client, client_lock=lock)

    with server.open_file_session(ctx, file_path, started=0.0) as session:
        assert isinstance(session, server.LeanFileSession)
        assert session.client is client
        assert session.rel_path == "Main.lean"
        assert session.identity is identity

    assert captured_calls["ctx"] is ctx
    assert captured_calls["path"] == file_path

    assert lock.acquire(blocking=False) is True
    lock.release()


def test_open_file_session_handles_missing_path(monkeypatch):
    monkeypatch.setattr(server, "setup_client_for_file", lambda ctx, path: None)

    ctx, _ = _make_ctx(client=SimpleNamespace(), client_lock=Lock())

    with pytest.raises(server.ToolError) as excinfo:
        with server.open_file_session(ctx, "/bad.lean", started=0.0):
            pass

    payload = excinfo.value.payload
    assert payload["isError"] is True
    structured = payload["structuredContent"]
    assert structured["code"] == server.ERROR_INVALID_PATH
    assert structured["details"]["file_path"] == "bad.lean"


def test_open_file_session_requires_client(monkeypatch):
    monkeypatch.setattr(server, "setup_client_for_file", lambda ctx, path: "Main.lean")
    identity = {"relative_path": "Main.lean", "uri": "file:///Main.lean"}
    monkeypatch.setattr(server, "_identity_for_rel_path", lambda ctx, rel: identity)

    ctx, _ = _make_ctx(client=None, client_lock=Lock())

    with pytest.raises(server.ToolError) as excinfo:
        with server.open_file_session(ctx, "/project/Main.lean", started=0.0):
            pass

    payload = excinfo.value.payload
    assert payload["structuredContent"]["code"] == server.ERROR_CLIENT_NOT_READY


def test_open_file_session_handles_missing_dependency(monkeypatch):
    message = "leanclient missing"

    def fake_setup(ctx, path):
        raise server.LeanclientNotInstalledError(message)

    monkeypatch.setattr(server, "setup_client_for_file", fake_setup)

    ctx, _ = _make_ctx(client=SimpleNamespace(), client_lock=Lock())

    with pytest.raises(server.ToolError) as excinfo:
        with server.open_file_session(ctx, "/project/Main.lean", started=0.0):
            pass

    payload = excinfo.value.payload
    structured = payload["structuredContent"]
    assert structured["code"] == server.ERROR_CLIENT_NOT_READY
    assert structured["details"]["dependency"] == "leanclient"
    assert message in structured["message"]
