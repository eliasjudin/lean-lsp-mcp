from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Final, Sequence

__all__ = ["ensure_test_project"]

PROJECT_DIRNAME: Final[str] = "test_project"
LEAN_TOOLCHAIN: Final[str] = "leanprover/lean4:v4.25.0\n"

LAKEFILE_TOML: Final[str] = """name = \"McpTestProject\"
version = \"0.1.0\"
defaultTargets = [\"McpTestProject\"]

[[lean_lib]]
name = \"McpTestProject\"
"""

LIB_MAIN_LEAN: Final[str] = """abbrev sampleValue : Nat := 42
"""

EDITOR_TOOLS_LEAN: Final[str] = """def sampleValue : Nat := 42

theorem sampleTheorem : True := by
  trivial

def completionTest : Nat := Nat.su
"""

MISC_TOOLS_LEAN: Final[str] = """def miscValue : Nat := 0

def multiAttemptTarget : Nat := 0
"""

LAKE_UPDATE: Final[Sequence[str]] = ("lake", "update", "--keep-toolchain")
LAKE_BUILD_STEPS: Final[tuple[Sequence[str], ...]] = (("lake", "build"),)


def ensure_test_project(repo_root: Path) -> Path:
    project_root = repo_root / "tests" / PROJECT_DIRNAME
    project_root.mkdir(parents=True, exist_ok=True)

    _write_if_changed(project_root / "lean-toolchain", LEAN_TOOLCHAIN)
    _write_if_changed(project_root / "lakefile.toml", LAKEFILE_TOML)
    _write_if_changed(project_root / "McpTestProject.lean", LIB_MAIN_LEAN)
    _write_if_changed(project_root / "EditorTools.lean", EDITOR_TOOLS_LEAN)
    _write_if_changed(project_root / "MiscTools.lean", MISC_TOOLS_LEAN)

    # Optional heavyweight setup for environments that need prebuilt Lake artifacts.
    # Disabled by default to keep local/CI MCP tests fast and deterministic.
    if _prepare_lake_requested():
        _run_lake_steps(project_root)

    return project_root


def _write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.write_text(content)


def _prepare_lake_requested() -> bool:
    return os.environ.get("LEAN_E2E_PREPARE_LAKE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _run_lake_steps(project_root: Path) -> None:
    manifest_path = project_root / "lake-manifest.json"
    if not manifest_path.exists():
        try:
            subprocess.run(LAKE_UPDATE, cwd=project_root, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "`lake` executable is required for end-to-end tests"
            ) from exc

    for args in LAKE_BUILD_STEPS:
        try:
            subprocess.run(args, cwd=project_root, check=True)
        except (
            FileNotFoundError
        ) as exc:  # pragma: no cover - relies on user environment
            raise RuntimeError(
                "`lake` executable is required for end-to-end tests"
            ) from exc
