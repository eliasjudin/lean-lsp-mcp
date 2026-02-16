from __future__ import annotations

from pathlib import Path

import pytest

from lean_lsp_mcp.pathing import PathResolutionError, resolve_workspace_path


def test_resolve_workspace_path_allows_relative(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "A.lean"
    target.write_text("theorem t : True := by trivial\n")

    resolved = resolve_workspace_path(root, "A.lean")
    assert resolved == target.resolve()


def test_resolve_workspace_path_rejects_absolute(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    with pytest.raises(PathResolutionError):
        resolve_workspace_path(root, str((root / "A.lean").resolve()))


def test_resolve_workspace_path_rejects_traversal(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.lean"
    outside.write_text("theorem t : True := by trivial\n")

    with pytest.raises(PathResolutionError):
        resolve_workspace_path(root, "../outside.lean")


def test_resolve_workspace_path_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.lean"
    outside.write_text("theorem t : True := by trivial\n")
    (root / "link.lean").symlink_to(outside)

    with pytest.raises(PathResolutionError):
        resolve_workspace_path(root, "link.lean")
