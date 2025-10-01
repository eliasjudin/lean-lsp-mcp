import os
from threading import Lock

from mcp.server.fastmcp import Context
from mcp.server.fastmcp.utilities.logging import get_logger
from leanclient import LeanLSPClient

from lean_lsp_mcp.file_utils import get_relative_file_path
from lean_lsp_mcp.utils import StdoutToStderr


logger = get_logger(__name__)


def startup_client(ctx: Context):
    """Initialize the Lean LSP client if not already set up.

    Args:
        ctx (Context): Context object.
    """
    lifespan = ctx.request_context.lifespan_context
    client_lock: Lock | None = getattr(lifespan, "client_lock", None)
    if client_lock is None:
        client_lock = Lock()
        lifespan.client_lock = client_lock
    with client_lock:
        lean_project_path = lifespan.lean_project_path
        if lean_project_path is None:
            raise ValueError("lean project path is not set.")

        client: LeanLSPClient | None = lifespan.client

        if client is not None:
            if client.project_path == lean_project_path:
                return
            client.close()
            lifespan.file_content_hashes.clear()

        with StdoutToStderr():
            try:
                client = LeanLSPClient(
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
                client = LeanLSPClient(
                    lean_project_path, initial_build=True, print_warnings=False
                )
                logger.info(
                    "Connected to Lean language server after build at %s",
                    lean_project_path,
                )
        lifespan.client = client


def valid_lean_project_path(path: str) -> bool:
    """Check if the given path is a valid Lean project path (contains a lean-toolchain file).

    Args:
        path (str): Absolute path to check.

    Returns:
        bool: True if valid Lean project path, False otherwise.
    """
    if not os.path.exists(path):
        return False
    return os.path.isfile(os.path.join(path, "lean-toolchain"))


def setup_client_for_file(ctx: Context, file_path: str) -> str | None:
    """Check if the current LSP client is already set up and correct for this file. Otherwise, set it up.

    Args:
        ctx (Context): Context object.
        file_path (str): Absolute path to the Lean file.

    Returns:
        str: Relative file path if the client is set up correctly, otherwise None.
    """
    lifespan = ctx.request_context.lifespan_context
    project_cache = getattr(lifespan, "project_cache", {})
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
            project_cache[current_dir] = ""

        prev_dir = current_dir
        current_dir = os.path.dirname(current_dir)

    return None
