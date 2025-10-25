from __future__ import annotations

import hashlib
import os
from threading import Lock
from typing import Dict, Optional, TYPE_CHECKING

from mcp.server.fastmcp import Context

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from leanclient import LeanLSPClient


def get_relative_file_path(lean_project_path: str, file_path: str) -> Optional[str]:
    """Convert a path to project-relative if it stays within the project root."""

    project_root = os.path.abspath(lean_project_path)
    candidates = []

    if os.path.exists(file_path):
        candidates.append(os.path.abspath(file_path))

    project_candidate = os.path.join(project_root, file_path)
    if os.path.exists(project_candidate):
        candidates.append(os.path.abspath(project_candidate))

    cwd_candidate = os.path.join(os.getcwd().strip(), file_path)
    if os.path.exists(cwd_candidate):
        candidates.append(os.path.abspath(cwd_candidate))

    for abs_path in candidates:
        try:
            if os.path.commonpath([abs_path, project_root]) == project_root:
                return os.path.relpath(abs_path, project_root)
        except ValueError:
            continue

    return None


def get_file_contents(abs_path: str) -> str:
    for enc in ("utf-8", "latin-1", None):
        try:
            if enc is None:
                with open(abs_path, "r") as f:
                    return f.read()
            with open(abs_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            if enc is None:
                raise
            continue


def update_file(ctx: Context, rel_path: str) -> str:
    """Update the file contents in the context.
    Args:
        ctx (Context): Context object.
        rel_path (str): Relative file path.

    Returns:
        str: Updated file contents.
    """
    # Get file contents and hash
    abs_path = os.path.join(
        ctx.request_context.lifespan_context.lean_project_path, rel_path
    )
    file_content = get_file_contents(abs_path)
    file_hash = hashlib.sha256(
        file_content.encode("utf-8", "surrogatepass")
    ).hexdigest()

    # Check if file_contents have changed
    lifespan = ctx.request_context.lifespan_context
    file_cache_lock: Lock | None = getattr(lifespan, "file_cache_lock", None)
    if file_cache_lock is None:
        file_cache_lock = Lock()
        lifespan.file_cache_lock = file_cache_lock

    with file_cache_lock:
        file_content_hashes: Dict[str, str] = lifespan.file_content_hashes
        previous_hash = file_content_hashes.get(rel_path)
        if previous_hash is None:
            file_content_hashes[rel_path] = file_hash
            return file_content
        if previous_hash == file_hash:
            return file_content
        file_content_hashes[rel_path] = file_hash

    client: LeanLSPClient = lifespan.client
    try:
        client.close_files([rel_path])
    except Exception:
        pass
    return file_content
