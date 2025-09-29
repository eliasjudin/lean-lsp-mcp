import os
from typing import Optional, Dict

from mcp.server.fastmcp import Context
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
    for enc in ("utf-8", "latin-1"):
        try:
            with open(abs_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(abs_path, "r", encoding=None) as f:
        return f.read()


def resolve_absolute_file_path(ctx: Context, file_path: str) -> Optional[str]:
    """Resolve ``file_path`` to an absolute path using the current context.

    This accepts absolute paths, project-relative paths and paths relative to the
    current working directory. It also consults any cached project roots the
    server discovered while handling previous requests.
    """

    normalized_path = os.path.expanduser(file_path.strip())
    if not normalized_path:
        return None

    candidates: list[str] = [normalized_path]

    lifespan = getattr(getattr(ctx, "request_context", None), "lifespan_context", None)
    if lifespan is not None:
        lean_project_path = getattr(lifespan, "lean_project_path", None)
        if lean_project_path:
            candidates.append(os.path.join(lean_project_path, normalized_path))
        project_cache = getattr(lifespan, "project_cache", None) or {}
        if project_cache and not os.path.isabs(normalized_path):
            for root in project_cache.values():
                if root:
                    candidates.append(os.path.join(root, normalized_path))

    if not os.path.isabs(normalized_path):
        candidates.append(os.path.join(os.getcwd(), normalized_path))

    seen: set[str] = set()
    for candidate in candidates:
        abs_candidate = os.path.abspath(candidate)
        if abs_candidate in seen:
            continue
        seen.add(abs_candidate)
        if os.path.exists(abs_candidate):
            return abs_candidate
    return None


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
    hashed_file = hash(file_content)

    # Check if file_contents have changed
    file_content_hashes: Dict[str, str] = (
        ctx.request_context.lifespan_context.file_content_hashes
    )
    if rel_path not in file_content_hashes:
        file_content_hashes[rel_path] = hashed_file
        return file_content

    elif hashed_file == file_content_hashes[rel_path]:
        return file_content

    # Update file_contents
    file_content_hashes[rel_path] = hashed_file

    # Reload file in LSP
    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    try:
        client.close_files([rel_path])
    except Exception:
        pass
    return file_content
