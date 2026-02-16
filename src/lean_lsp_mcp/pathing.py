from __future__ import annotations

from pathlib import Path


class PathResolutionError(ValueError):
    """Raised when a client-provided path is outside workspace boundaries."""


def resolve_workspace_path(workspace_root: Path, relative_path: str) -> Path:
    """Resolve a user-provided relative path under workspace root safely.

    This rejects absolute paths and path traversal outside the workspace root,
    even through symlinks.
    """
    if not relative_path or not relative_path.strip():
        raise PathResolutionError("Path must not be empty.")

    raw = Path(relative_path)
    if raw.is_absolute():
        raise PathResolutionError(
            "Absolute paths are not allowed; use project-relative paths."
        )

    normalized = Path(str(raw).replace("\\", "/"))
    if any(part == ".." for part in normalized.parts):
        raise PathResolutionError("Path traversal is not allowed.")

    root = workspace_root.resolve()
    candidate = (root / normalized).resolve()

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PathResolutionError("Path escapes workspace root.") from exc

    return candidate


def to_workspace_relative(workspace_root: Path, absolute_path: Path) -> str:
    """Convert absolute path to workspace-relative path string."""
    root = workspace_root.resolve()
    resolved = absolute_path.resolve()
    try:
        rel = resolved.relative_to(root)
    except ValueError as exc:
        raise PathResolutionError("Resolved path is outside workspace root.") from exc
    return str(rel)
