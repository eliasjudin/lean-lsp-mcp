import os
from threading import Lock
from types import SimpleNamespace

import pytest

from conftest import load_from_src


def _make_ctx(
    *,
    lean_project_path=None,
    client=None,
    file_content_hashes=None,
    project_cache=None,
    client_lock=None,
    file_cache_lock=None,
    project_cache_lock=None,
):
    lifespan = SimpleNamespace(
        lean_project_path=lean_project_path,
        client=client,
        file_content_hashes=file_content_hashes if file_content_hashes is not None else {},
        project_cache=project_cache if project_cache is not None else {},
        client_lock=client_lock,
        file_cache_lock=file_cache_lock,
        project_cache_lock=project_cache_lock,
    )
    ctx = SimpleNamespace(
        request_context=SimpleNamespace(lifespan_context=lifespan)
    )
    return ctx, lifespan


def test_setup_client_revalidates_negative_cache(tmp_path):
    client_utils = load_from_src("lean_lsp_mcp.client_utils")

    project_root = tmp_path / "proj"
    project_root.mkdir()
    src_dir = project_root / "src"
    src_dir.mkdir()
    lean_file = src_dir / "Example.lean"
    lean_file.write_text("-- test")

    ctx, lifespan = _make_ctx(
        client_lock=Lock(),
    )

    first_result = client_utils.setup_client_for_file(ctx, str(lean_file))
    assert first_result is None
    assert lifespan.project_cache[str(src_dir)] == ""
    assert lifespan.project_cache[str(project_root)] == ""

    (project_root / "lean-toolchain").write_text("")

    second_result = client_utils.setup_client_for_file(ctx, str(lean_file))

    assert second_result == os.path.join("src", "Example.lean")
    assert lifespan.lean_project_path == str(project_root)
    assert lifespan.project_cache[str(src_dir)] == str(project_root)
    assert lifespan.project_cache[str(project_root)] == str(project_root)


def test_startup_client_requires_project_path(monkeypatch):
    client_utils = load_from_src("lean_lsp_mcp.client_utils")

    class DummyClient:
        def __init__(self, *_args, **_kwargs):
            self.project_path = ""

        def close(self):
            pass

    monkeypatch.setattr(
        client_utils,
        "ensure_leanclient_available",
        lambda: (DummyClient, object),
    )

    ctx, _ = _make_ctx()

    with pytest.raises(ValueError):
        client_utils.startup_client(ctx)


def test_startup_client_reinitializes_on_project_change(monkeypatch, tmp_path):
    client_utils = load_from_src("lean_lsp_mcp.client_utils")

    class DummyClient:
        instances = []

        def __init__(self, project_path, *, initial_build, print_warnings):
            self.project_path = project_path
            self.closed = False
            DummyClient.instances.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        client_utils,
        "ensure_leanclient_available",
        lambda: (DummyClient, object),
    )

    existing_client = DummyClient(str(tmp_path / "old"), initial_build=True, print_warnings=False)
    cache = {"foo": "bar"}
    ctx, lifespan = _make_ctx(
        lean_project_path=str(tmp_path / "new"),
        client=existing_client,
        file_content_hashes=cache,
        client_lock=Lock(),
        file_cache_lock=Lock(),
    )

    client_utils.startup_client(ctx)

    assert existing_client.closed is True
    assert lifespan.file_content_hashes == {}
    assert cache == {}
    assert lifespan.client is DummyClient.instances[-1]
    assert lifespan.client is not existing_client
    assert lifespan.client.project_path == str(tmp_path / "new")


def test_startup_client_retries_with_initial_build(monkeypatch, tmp_path):
    client_utils = load_from_src("lean_lsp_mcp.client_utils")

    call_flags: list[bool] = []

    class RetryClient:
        def __init__(self, project_path, *, initial_build, print_warnings):
            call_flags.append(initial_build)
            if not initial_build:
                raise RuntimeError("first attempt fails")
            self.project_path = project_path

        def close(self):
            pass

    monkeypatch.setattr(
        client_utils,
        "ensure_leanclient_available",
        lambda: (RetryClient, object),
    )

    ctx, lifespan = _make_ctx(
        lean_project_path=str(tmp_path / "proj"),
        client=None,
        client_lock=Lock(),
        file_cache_lock=Lock(),
    )

    client_utils.startup_client(ctx)

    assert call_flags == [False, True]
    assert lifespan.client.project_path == str(tmp_path / "proj")
