from __future__ import annotations

import os
from threading import Lock
from typing import TYPE_CHECKING

from mcp.server.fastmcp import Context
from mcp.server.fastmcp.utilities.logging import get_logger

from lean_lsp_mcp.file_utils import get_relative_file_path
from lean_lsp_mcp.leanclient_provider import ensure_leanclient_available
from lean_lsp_mcp.utils import StdoutToStderr

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from leanclient import LeanLSPClient


logger = get_logger(__name__)


def startup_client(ctx: Context):
    """Initialize and cache a Lean LSP client for the active project.

    Args:
        ctx: FastMCP context whose ``request_context`` tracks the Lean lifecycle state.

    Returns:
        None: The client instance is stored on the context lifespan for reuse.

    Raises:
        ValueError: If ``lifespan.lean_project_path`` has not been populated yet.
        LeanclientNotInstalledError: If the optional ``leanclient`` dependency cannot be imported.
    """
    lifespan = ctx.request_context.lifespan_context
    client_lock: Lock | None = getattr(lifespan, "client_lock", None)
    if client_lock is None:
        client_lock = Lock()
        lifespan.client_lock = client_lock
    file_cache_lock: Lock | None = getattr(lifespan, "file_cache_lock", None)
    if file_cache_lock is None:
        file_cache_lock = Lock()
        lifespan.file_cache_lock = file_cache_lock
    # Fail fast if the optional dependency is missing so callers can surface a
    # helpful tool error instead of crashing the server import.
    LeanClientCls, _ = ensure_leanclient_available()
    with client_lock:
        lean_project_path = lifespan.lean_project_path
        if lean_project_path is None:
            raise ValueError("lean project path is not set.")

        client: LeanLSPClient | None = lifespan.client

        if client is not None:
            if client.project_path == lean_project_path:
                return
            client.close()
            with file_cache_lock:
                lifespan.file_content_hashes.clear()

        with StdoutToStderr():
            try:
                client = LeanClientCls(
                    lean_project_path, initial_build=False, print_warnings=False
                )
                logger.info(
                    "Connected to Lean language server at %s", lean_project_path
                )
            except Exception as e:
                logger.warning(
                    "Lean LSP startup without build failed, retrying with initial build: %s",
                    e,
                )
                client = LeanClientCls(
                    lean_project_path, initial_build=True, print_warnings=False
                )
                logger.info(
                    "Connected to Lean language server after build at %s",
                    lean_project_path,
                )
        lifespan.client = client


def valid_lean_project_path(path: str) -> bool:
    """Return whether ``path`` points to a Lean project root containing ``lean-toolchain``.

    Args:
        path: Absolute directory candidate to inspect.

    Returns:
        bool: ``True`` when the directory exists and includes ``lean-toolchain``; ``False`` otherwise.
    """
    if not os.path.exists(path):
        return False
    return os.path.isfile(os.path.join(path, "lean-toolchain"))


def setup_client_for_file(ctx: Context, file_path: str) -> str | None:
    """Ensure the cached Lean client is configured for ``file_path`` and return its project-relative path.

    Args:
        ctx: FastMCP context coordinating Lean project state.
        file_path: Absolute path to the Lean source file requiring synchronization.

    Returns:
        str | None: Project-relative path when the client is ready for the file; ``None`` if the file cannot be
            associated with a Lean project root.

    Raises:
        ValueError: Propagated from :func:`startup_client` when the Lean project path is unset.

    Examples:
        >>> rel_path = setup_client_for_file(ctx, "/abs/project/Main.lean")
        >>> rel_path.endswith(".lean")
        True
    """
    lifespan = ctx.request_context.lifespan_context
    project_cache = getattr(lifespan, "project_cache", None)
    if project_cache is None:
        project_cache = {}
        lifespan.project_cache = project_cache
    project_cache_lock: Lock | None = getattr(lifespan, "project_cache_lock", None)
    if project_cache_lock is None:
        project_cache_lock = Lock()
        lifespan.project_cache_lock = project_cache_lock
    abs_file_path = os.path.abspath(file_path)
    file_dir = os.path.dirname(abs_file_path)

    def activate_project(project_path: str, cache_dirs: list[str]) -> str | None:
        rel = get_relative_file_path(project_path, file_path)
        if rel is None:
            return None

        project_path = os.path.abspath(project_path)
        lifespan.lean_project_path = project_path

        cache_targets: list[str] = []
        for directory in cache_dirs + [project_path]:
            if directory and directory not in cache_targets:
                cache_targets.append(directory)

        with project_cache_lock:
            for directory in cache_targets:
                project_cache[directory] = project_path
        startup_client(ctx)
        return rel

    # Check if the file_path works for the current lean_project_path.
    lean_project_path = lifespan.lean_project_path
    if lean_project_path is not None:
        rel_path = activate_project(
            lean_project_path,
            [file_dir],
        )
        if rel_path is not None:
            return rel_path

    # Try to find the new correct project path by checking all directories in file_path.
    prev_dir = None
    visited_dirs: list[str] = []
    current_dir = file_dir

    while current_dir and current_dir != prev_dir:
        visited_dirs.append(current_dir)
        with project_cache_lock:
            cached_root = project_cache.get(current_dir)
        if cached_root:
            rel_path = activate_project(cached_root, visited_dirs)
            if rel_path is not None:
                return rel_path
        elif valid_lean_project_path(current_dir):
            rel_path = activate_project(current_dir, visited_dirs)
            if rel_path is not None:
                return rel_path
        else:
            with project_cache_lock:
                project_cache[current_dir] = ""

        prev_dir = current_dir
        current_dir = os.path.dirname(current_dir)

    return None
