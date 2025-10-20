from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from conftest import load_from_src

server = load_from_src("lean_lsp_mcp.server")


def _make_ctx(project_root: str | None = None) -> SimpleNamespace:
    lifespan = SimpleNamespace(
        lean_project_path=project_root,
        file_content_hashes={},
        client=None,
        client_lock=None,
        rate_limit={},
    )
    request_context = SimpleNamespace(lifespan_context=lifespan)
    return SimpleNamespace(request_context=request_context)


def test_file_contents_rejects_directory(tmp_path: Path) -> None:
    ctx = _make_ctx(project_root=str(tmp_path))
    params = server.LeanFileContentsInput(file_path=str(tmp_path))
    result = server.file_contents(ctx, params)

    assert result["isError"] is True
    structured = result["structuredContent"]
    assert structured["code"] == server.ERROR_INVALID_PATH
    assert structured["details"]["kind"] == "directory"
    # Text summary is the first content item by default
    assert "directory" in result["content"][0]["text"].lower()


def test_file_contents_missing_path(tmp_path: Path) -> None:
    ctx = _make_ctx(project_root=str(tmp_path))
    params = server.LeanFileContentsInput(file_path="missing.lean")
    result = server.file_contents(ctx, params)

    assert result["isError"] is True
    structured = result["structuredContent"]
    assert structured["code"] == server.ERROR_INVALID_PATH
    assert structured["details"]["path"] == "missing.lean"


def test_file_contents_reads_project_relative(tmp_path: Path) -> None:
    formalization = tmp_path / "Formalization"
    formalization.mkdir()
    file_path = formalization / "example.lean"
    file_path.write_text("import Mathlib\n\n#check Nat\n", encoding="utf-8")

    ctx = _make_ctx(project_root=str(tmp_path))
    params = server.LeanFileContentsInput(
        file_path="Formalization/example.lean",
        annotate_lines=True,
    )
    result = server.file_contents(ctx, params)

    assert result["isError"] is False
    structured = result["structuredContent"]

    assert structured["file"]["path"] == "Formalization/example.lean"
    assert structured["annotated"] is True
    lines = structured["lines"]
    assert lines[0]["text"] == "import Mathlib"
    assert lines[1]["text"] == ""
    assert lines[2]["text"] == "#check Nat"
    # Summary appears as the first content item
    summary = result["content"][0]["text"]
    assert summary
    assert "/" in summary
