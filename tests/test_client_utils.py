import os
from threading import Lock
from types import SimpleNamespace

from conftest import load_from_src


def test_setup_client_revalidates_negative_cache(tmp_path):
    client_utils = load_from_src("lean_lsp_mcp.client_utils")

    project_root = tmp_path / "proj"
    project_root.mkdir()
    src_dir = project_root / "src"
    src_dir.mkdir()
    lean_file = src_dir / "Example.lean"
    lean_file.write_text("-- test")

    lifespan = SimpleNamespace(
        lean_project_path=None,
        client=None,
        file_content_hashes={},
        project_cache={},
        client_lock=Lock(),
    )
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=lifespan))

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
