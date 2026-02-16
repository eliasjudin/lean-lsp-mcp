from __future__ import annotations

from pathlib import Path


def test_lean_lsp_mcp_import_origin(repo_root: Path) -> None:
    import lean_lsp_mcp  # noqa: PLC0415

    module_path = Path(lean_lsp_mcp.__file__).resolve()
    assert str(module_path).startswith(str((repo_root / "src").resolve())), (
        f"lean_lsp_mcp imported from outside this repository: {module_path}"
    )
