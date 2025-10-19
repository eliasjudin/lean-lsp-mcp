from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from contextlib import asynccontextmanager, contextmanager
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import urllib
import json
import functools
import subprocess
import uuid
from threading import Lock
from types import SimpleNamespace

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger
from mcp.server.auth.settings import AuthSettings

from lean_lsp_mcp.client_utils import setup_client_for_file, startup_client
from lean_lsp_mcp.file_utils import (
    get_file_contents,
    get_relative_file_path,
    update_file,
)
from lean_lsp_mcp.instructions import INSTRUCTIONS
from lean_lsp_mcp.tool_spec import build_tool_spec
from lean_lsp_mcp.schema import mcp_result
from lean_lsp_mcp.schema_types import (
    ERROR_BAD_REQUEST,
    ERROR_CLIENT_NOT_READY,
    ERROR_INVALID_PATH,
    ERROR_IO_FAILURE,
    ERROR_NO_GOAL,
    ERROR_NOT_GOAL_POSITION,
    ERROR_RATE_LIMIT,
    ERROR_UNKNOWN,
    FileIdentity,
)
from lean_lsp_mcp.utils import (
    OutputCapture,
    compute_pagination,
    diagnostics_to_entries,
    extract_range,
    file_identity,
    filter_diagnostics_by_position,
    find_start_position,
    format_diagnostics,
    format_goal,
    format_line,
    goal_to_payload,
    normalize_range,
    OptionalTokenVerifier,
    summarize_diagnostics,
    uri_to_absolute_path,
)
from lean_lsp_mcp.leanclient_provider import (
    LeanclientNotInstalledError,
    ensure_leanclient_available,
    is_leanclient_available,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for typing only
    from leanclient import DocumentContentChange, LeanLSPClient


try:  # pragma: no cover - metadata lookup may fail in tests
    SERVER_VERSION = version("lean-lsp-mcp")
except PackageNotFoundError:  # pragma: no cover - local dev fallback
    SERVER_VERSION = None


logger = get_logger(__name__)


# Server and context
@dataclass
class AppContext:
    lean_project_path: str | None
    client: LeanLSPClient | None
    file_content_hashes: Dict[str, str]
    rate_limit: Dict[str, List[int]]
    project_cache: Dict[str, str]
    client_lock: Lock


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    try:
        lean_project_path = os.environ.get("LEAN_PROJECT_PATH", "").strip()
        if not lean_project_path:
            lean_project_path = None
        else:
            lean_project_path = os.path.abspath(lean_project_path)

        context = AppContext(
            lean_project_path=lean_project_path,
            client=None,
            file_content_hashes={},
            rate_limit={
                "leansearch": [],
                "loogle": [],
                "lean_state_search": [],
                "hammer_premise": [],
            },
            project_cache={},
            client_lock=Lock(),
        )
        if context.lean_project_path:
            try:
                logger.info(
                    "Prewarming Lean client for %s", context.lean_project_path
                )
                dummy_ctx = SimpleNamespace(
                    request_context=SimpleNamespace(lifespan_context=context)
                )
                startup_client(dummy_ctx)
            except Exception as exc:  # pragma: no cover - prewarm best effort
                logger.warning(
                    "Lean client prewarm failed for %s: %s",
                    context.lean_project_path,
                    exc,
                )
        yield context
    finally:
        logger.info("Closing Lean LSP client")
        if context.client:
            context.client.close()


mcp_kwargs = dict(
    name="Lean LSP",
    instructions=INSTRUCTIONS,
    lifespan=app_lifespan,
)
if is_leanclient_available():
    mcp_kwargs["dependencies"] = ["leanclient"]

auth_token = os.environ.get("LEAN_LSP_MCP_TOKEN")
if auth_token:
    mcp_kwargs["auth"] = AuthSettings(
        type="optional",
        issuer_url="http://localhost/dummy-issuer",
        resource_server_url="http://localhost/dummy-resource",
    )
    mcp_kwargs["token_verifier"] = OptionalTokenVerifier(auth_token)

mcp = FastMCP(**mcp_kwargs)


@contextmanager
def client_session(ctx: Context):
    lifespan = ctx.request_context.lifespan_context
    lock = getattr(lifespan, "client_lock", None)
    if lock is None:
        yield lifespan.client
        return
    lock.acquire()
    try:
        yield lifespan.client
    finally:
        lock.release()


class ToolError(Exception):
    """Internal control-flow exception carrying a ready-to-send MCP response."""

    def __init__(self, payload: Dict[str, Any]):
        super().__init__(payload.get("structured", {}).get("message", "Tool error"))
        self.payload = payload


@dataclass
class LeanFileSession:
    """State bundle for working with a Lean file while the client lock is held."""

    ctx: Context
    client: LeanLSPClient
    rel_path: str
    identity: FileIdentity
    _content: str | None = None

    def load_content(self, *, refresh: bool = False) -> str:
        """Load and cache the current file contents via the LSP client."""

        if refresh or self._content is None:
            self._content = update_file(self.ctx, self.rel_path)
        return self._content

    def clear_cache(self) -> None:
        """Drop the cached content if external updates were applied."""

        self._content = None


def _normalize_format(value: Optional[str], *, default: str = "compact") -> str:
    """Formats are now fixed to compact; keep helper for backward compatibility."""

    return "compact"


def _compact_pos(*, line: int | None = None, column: int | None = None) -> Dict[str, int]:
    """Return compact position payloads using 1-based indices."""

    pos: Dict[str, int] = {}
    if line is not None:
        pos["l"] = max(line, 1)
    if column is not None:
        pos["c"] = max(column, 1)
    return pos


@contextmanager
def open_file_session(
    ctx: Context,
    file_path: str,
    *,
    started: float,
    line: int | None = None,
    column: int | None = None,
    category: str | None = None,
    invalid_details: Dict[str, Any] | None = None,
    client_details: Dict[str, Any] | None = None,
    invalid_message: str = "Invalid Lean file path: Unable to start LSP server or load file",
    client_message: str = (
        "Lean client is not available. Run another tool to initialize the project first."
    ),
) -> Iterator[LeanFileSession]:
    """Context manager yielding a `LeanFileSession` or raising `ToolError`.

    Centralizes project detection, client acquisition, and consistent error payloads.
    """

    invalid_payload = dict(invalid_details or {})
    invalid_payload.setdefault("file_path", _sanitize_path_label(file_path))
    if line is not None:
        invalid_payload.setdefault("line", line)
    if column is not None:
        invalid_payload.setdefault("column", column)

    try:
        rel_path = setup_client_for_file(ctx, file_path)
    except LeanclientNotInstalledError as exc:
        dependency_payload = dict(invalid_payload)
        dependency_payload.setdefault("dependency", "leanclient")
        raise ToolError(
            error_result(
                message=str(exc),
                code=ERROR_CLIENT_NOT_READY,
                category=category,
                details=dependency_payload,
                start_time=started,
                ctx=ctx,
            )
        )
    if not rel_path:
        raise ToolError(
            error_result(
                message=invalid_message,
                code=ERROR_INVALID_PATH,
                category=category,
                details=invalid_payload,
                start_time=started,
                ctx=ctx,
            )
        )

    identity = _identity_for_rel_path(ctx, rel_path)
    ready_payload = dict(client_details or {})
    ready_payload.setdefault("file", identity["relative_path"])
    if line is not None:
        ready_payload.setdefault("line", line)
    if column is not None:
        ready_payload.setdefault("column", column)

    with client_session(ctx) as client:
        if client is None:
            raise ToolError(
                error_result(
                    message=client_message,
                    code=ERROR_CLIENT_NOT_READY,
                    category=category,
                    details=ready_payload,
                    start_time=started,
                    ctx=ctx,
                )
            )

        yield LeanFileSession(
            ctx=ctx,
            client=client,
            rel_path=rel_path,
            identity=identity,
        )


def _text_item(text: str) -> Dict[str, str]:
    return {"type": "text", "text": text}


def _resource_item(uri: str, text: str, mime_type: str = "text/plain") -> Dict[str, Any]:
    return {
        "type": "resource",
        "resource": {"uri": uri, "mimeType": mime_type, "text": text},
    }


def _json_item(structured: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON resource content item that mirrors structuredContent.

    This is placed first in the content array so generic MCP clients can
    reliably access machine-readable results without special handling.
    """
    try:
        json_text = json.dumps(structured, ensure_ascii=False)
    except Exception:
        # As a hard fallback, present an empty object to avoid breaking the
        # invariant that content[0] is a JSON resource when structured exists.
        json_text = "{}"
    return _resource_item(
        f"file:///_mcp_structured_{uuid.uuid4().hex}.json",
        json_text,
        mime_type="application/json",
    )


def _extract_request_id(ctx: Context | None) -> str | None:
    """Best-effort extraction of the MCP request identifier."""

    if ctx is None:
        return None

    candidates = []
    for attr in ("request_id", "requestId", "requestID", "id"):
        value = getattr(ctx, attr, None)
        if value:
            return str(value)

    request_context = getattr(ctx, "request_context", None)
    if request_context is None:
        return None

    for attr in ("request_id", "requestId", "requestID", "id"):
        value = getattr(request_context, attr, None)
        if value:
            return str(value)

    rpc_request = getattr(request_context, "rpc_request", None)
    if rpc_request is not None:
        for attr in ("request_id", "requestId", "requestID", "id"):
            value = getattr(rpc_request, attr, None)
            if value:
                return str(value)

    rpc_request_id = getattr(request_context, "rpc_request_id", None)
    if rpc_request_id:
        return str(rpc_request_id)

    return None


def _build_meta(ctx: Context | None, start_time: float) -> Dict[str, Any]:
    """Construct the `_meta` payload for tool responses."""

    duration_ms = max(int((time.perf_counter() - start_time) * 1000), 0)
    meta: Dict[str, Any] = {"duration_ms": duration_ms}

    request_id = _extract_request_id(ctx)
    if request_id:
        meta["request_id"] = request_id

    return meta


def success_result(
    *,
    summary: str,
    structured: Dict[str, Any] | None,
    start_time: float,
    ctx: Context | None = None,
    content: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    # Machine-first: if we have structured content, expose it as the first
    # content item (JSON resource). Append caller-provided content items next,
    # or (if none provided) a human-readable summary.
    payload: List[Dict[str, Any]] = []
    if structured is not None:
        payload.append(_json_item(structured))
    if content:
        payload.extend(content)
    else:
        payload.append(_text_item(summary))
    return mcp_result(
        content=payload,
        structured=structured,
        meta=_build_meta(ctx, start_time),
    )


def error_result(
    *,
    message: str,
    start_time: float,
    ctx: Context | None = None,
    code: str | None = None,
    category: str | None = None,
    details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    structured: Dict[str, Any] = {"message": message}
    if code:
        structured["code"] = code
    if category:
        structured["category"] = category
    if details:
        structured["details"] = details
    content: List[Dict[str, Any]] = [_json_item(structured), _text_item(message)]
    return mcp_result(
        content=content,
        structured=structured,
        is_error=True,
        meta=_build_meta(ctx, start_time),
    )


def _sanitize_path_label(path: str) -> str:
    if not path:
        return path
    try:
        rel = os.path.relpath(path, os.getcwd())
        if not rel.startswith(".."):
            return rel.replace(os.sep, "/")
    except ValueError:  # pragma: no cover - handles different drive on Windows
        pass
    basename = os.path.basename(path)
    return basename if basename else path


def _identity_for_rel_path(ctx: Context, rel_path: str) -> FileIdentity:
    project_root = ctx.request_context.lifespan_context.lean_project_path
    absolute = None
    if project_root:
        absolute = os.path.join(project_root, rel_path)
    elif os.path.isabs(rel_path):
        absolute = rel_path
    return file_identity(rel_path, absolute)

# Rate limiting: n requests per m seconds
def rate_limited(category: str, max_requests: int, per_seconds: int):
    def decorator(func):
        docstring = func.__doc__ or ""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            ctx = kwargs.get("ctx")
            if ctx is None:
                if args:
                    ctx = args[0]
                else:  # pragma: no cover - defensive guard for misconfigured tools
                    raise TypeError("rate-limited tool requires a Context as the first argument")

            rate_limit = ctx.request_context.lifespan_context.rate_limit
            timestamps = rate_limit.setdefault(category, [])
            current_time = int(time.time())
            rate_limit[category] = [
                timestamp
                for timestamp in timestamps
                if timestamp > current_time - per_seconds
            ]
            if len(rate_limit[category]) >= max_requests:
                message = (
                    f"Tool limit exceeded: {max_requests} requests per {per_seconds} s."
                    " Try again later."
                )
                return error_result(
                    message=message,
                    code=ERROR_RATE_LIMIT,
                    category=category,
                    details={"max_requests": max_requests, "per_seconds": per_seconds},
                    start_time=started,
                    ctx=ctx,
                )
            rate_limit[category].append(current_time)
            return func(*args, **kwargs)

        limit_prefix = f"Limit: {max_requests}req/{per_seconds}s."
        wrapper.__doc__ = (
            f"{limit_prefix} {docstring}" if docstring else limit_prefix
        )
        return wrapper

    return decorator


# Project level tools
@mcp.tool("lean_build")
def lsp_build(ctx: Context, lean_project_path: str = None, clean: bool = False, _format: Optional[str] = None) -> Any:
    """Build the Lean project and restart the LSP Server.

    Use only if needed (e.g. new imports).

    Args:
        lean_project_path (str, optional): Path to the Lean project. If not provided, it will be inferred from previous tool calls.
        clean (bool, optional): Run `lake clean` before building. Attention: Only use if it is really necessary! It can take a long time! Defaults to False.

    Returns:
        str: Build output or error msg
    """
    started = time.perf_counter()
    if not lean_project_path:
        lean_project_path = ctx.request_context.lifespan_context.lean_project_path
    else:
        lean_project_path = os.path.abspath(lean_project_path)
        ctx.request_context.lifespan_context.lean_project_path = lean_project_path

    if not lean_project_path:
        message = (
            "No Lean project path configured. Provide `lean_project_path` or set "
            "`LEAN_PROJECT_PATH`."
        )
        return error_result(
            message=message,
            code=ERROR_BAD_REQUEST,
            start_time=started,
            ctx=ctx,
        )

    build_output = ""
    build_output_parts: List[str] = []
    sanitized_project = _sanitize_path_label(lean_project_path)
    project_uri = Path(lean_project_path).resolve().as_uri()

    def _record_output(text: str | None) -> None:
        if text:
            build_output_parts.append(text)

    lifespan = ctx.request_context.lifespan_context
    try:
        client: LeanLSPClient | None = lifespan.client
        if client:
            client.close()
            lifespan.file_content_hashes.clear()

        if clean:
            clean_proc = subprocess.run(
                ["lake", "clean"],
                cwd=lean_project_path,
                check=False,
                capture_output=True,
                text=True,
            )
            _record_output(clean_proc.stdout)
            _record_output(clean_proc.stderr)
            logger.info("Ran `lake clean`")

        if is_leanclient_available():
            LeanClientCls, _ = ensure_leanclient_available()
            with OutputCapture() as output:
                client = LeanClientCls(
                    lean_project_path,
                    initial_build=True,
                    print_warnings=False,
                )
            lifespan.client = client
            logger.info("Built project and re-started LSP client")
            _record_output(output.get_output())
            build_output = "".join(build_output_parts)
            summary = f"Build ok (clean={str(clean).lower()})"
            structured = {
                "project": {"path": sanitized_project},
                "clean": clean,
                "status": "ok",
                "lsp_restarted": True,
            }
            content_items = [_text_item(summary)]
            if build_output and len(build_output) <= 4000:
                content_items.append(_resource_item(project_uri, build_output))
            return success_result(
                summary=summary,
                structured=structured,
                start_time=started,
                ctx=ctx,
                content=content_items,
            )

        # Fallback path: leanclient not installed; run `lake build` directly.
        build_proc = subprocess.run(
            ["lake", "build"],
            cwd=lean_project_path,
            check=False,
            capture_output=True,
            text=True,
        )
        _record_output(build_proc.stdout)
        _record_output(build_proc.stderr)
        build_output = "".join(build_output_parts)
        if build_proc.returncode != 0:
            message = "`lake build` failed"
            details = {
                "project_path": sanitized_project,
                "clean": clean,
                "exit_code": build_proc.returncode,
                "output": build_output,
            }
            return error_result(
                message=message,
                code=ERROR_IO_FAILURE,
                details=details,
                start_time=started,
                ctx=ctx,
            )

        lifespan.client = None
        summary = f"Build ok (clean={str(clean).lower()}, leanclient missing)"
        structured = {
            "project": {"path": sanitized_project},
            "clean": clean,
            "status": "ok",
            "lsp_restarted": False,
        }
        content_items = [_text_item(summary)]
        if build_output and len(build_output) <= 4000:
            content_items.append(_resource_item(project_uri, build_output))
        return success_result(
            summary=summary,
            structured=structured,
            start_time=started,
            ctx=ctx,
            content=content_items,
        )
    except Exception as exc:
        if build_output_parts:
            build_output = "".join(build_output_parts)
        message = f"Build failed: {exc}"
        details = {
            "project_path": sanitized_project,
            "clean": clean,
            "output": build_output,
        }
        return error_result(
            message=message,
            code=ERROR_IO_FAILURE,
            details=details,
            start_time=started,
            ctx=ctx,
        )


# File level tools
@mcp.tool("lean_file_contents")
def file_contents(
    ctx: Context,
    file_path: str,
    annotate_lines: bool = True,
    start_line: Optional[int] = None,
    line_count: Optional[int] = None,
    _format: Optional[str] = None,
) -> Any:
    """Get the text contents of a Lean file, optionally with line numbers.

    Args:
        file_path (str): Path to the Lean file. Absolute paths are recommended. Relative
            paths are resolved against the configured Lean project root (if available)
            or else the current working directory.
        annotate_lines (bool, optional): Annotate lines with line numbers. Defaults to True.
        start_line (int, optional): 1-based line to start from.
        line_count (int, optional): Number of lines to return from ``start_line``.

    Returns:
        str: File content or error msg
    """
    started = time.perf_counter()
    sanitized_path = _sanitize_path_label(file_path)
    lifespan = ctx.request_context.lifespan_context
    project_root = getattr(lifespan, "lean_project_path", None)
    if not project_root:
        client = getattr(lifespan, "client", None)
        if client is not None:
            project_root = getattr(client, "project_path", None)

    expanded = os.path.expanduser(file_path)
    candidates: List[str] = []

    if expanded:
        if os.path.isabs(expanded):
            candidates.append(expanded)
        else:
            if project_root:
                candidates.append(os.path.join(project_root, expanded))
            candidates.append(os.path.join(os.getcwd(), expanded))

    # Fallback: if all else fails, try the raw input as-is (handles empty strings gracefully).
    if expanded and expanded not in candidates:
        candidates.append(expanded)
    elif not expanded:
        candidates.append(expanded)

    resolved_path = next((path for path in candidates if os.path.exists(path)), None)

    if resolved_path is None:
        # Provide a more actionable message when a relative path is given but no
        # project root is configured (and the CWD candidate also failed).
        is_relative = bool(expanded) and not os.path.isabs(expanded)
        project_label = _sanitize_path_label(project_root) if project_root else None

        if is_relative and not project_root:
            message = (
                f"File `{sanitized_path}` does not exist relative to the current directory, "
                "and no Lean project root is configured. Provide an absolute path, or set "
                "`LEAN_PROJECT_PATH` / run `lean_build` for your project, then try again."
            )
        elif is_relative and project_root:
            message = (
                f"File `{sanitized_path}` was not found under the project root "
                f"`{project_label}`. Check the path or provide an absolute path."
            )
        else:
            message = (
                f"File `{sanitized_path}` does not exist. Please check the path and try again."
            )

        details: Dict[str, Any] = {"path": sanitized_path}
        # Include additional context to aid debugging without breaking existing consumers.
        if is_relative:
            details["project_root"] = project_label
            # Sanitize candidate paths to avoid leaking absolute host paths.
            details["candidates"] = [
                _sanitize_path_label(p) for p in candidates
            ]

        return error_result(
            message=message,
            code=ERROR_INVALID_PATH,
            details=details,
            start_time=started,
            ctx=ctx,
        )

    resolved_path = os.path.abspath(resolved_path)

    if os.path.isdir(resolved_path):
        message = f"Path `{sanitized_path}` is a directory. Provide a Lean source file."
        return error_result(
            message=message,
            code=ERROR_INVALID_PATH,
            details={"path": sanitized_path, "kind": "directory"},
            start_time=started,
            ctx=ctx,
        )

    try:
        data = get_file_contents(resolved_path)
    except FileNotFoundError:
        message = (
            f"File `{sanitized_path}` does not exist. Please check the path and try again."
        )
        return error_result(
            message=message,
            code=ERROR_INVALID_PATH,
            details={"path": sanitized_path},
            start_time=started,
            ctx=ctx,
        )
    except IsADirectoryError:
        message = f"Path `{sanitized_path}` is a directory. Provide a Lean source file."
        return error_result(
            message=message,
            code=ERROR_INVALID_PATH,
            details={"path": sanitized_path, "kind": "directory"},
            start_time=started,
            ctx=ctx,
        )

    if start_line is not None and start_line < 1:
        return error_result(
            message="`start_line` must be >= 1",
            code=ERROR_BAD_REQUEST,
            details={"start_line": start_line},
            start_time=started,
            ctx=ctx,
        )
    if line_count is not None and line_count < 1:
        return error_result(
            message="`line_count` must be >= 1",
            code=ERROR_BAD_REQUEST,
            details={"line_count": line_count},
            start_time=started,
            ctx=ctx,
        )

    rel_path = None
    if project_root:
        try:
            rel_path = get_relative_file_path(project_root, resolved_path)
        except Exception:
            rel_path = None
    display_label = rel_path if rel_path else resolved_path
    identity = file_identity(display_label, resolved_path)

    lines = data.split("\n")
    total_lines = len(lines)

    if start_line and total_lines and start_line > total_lines:
        return error_result(
            message="`start_line` is beyond the end of the file",
            code=ERROR_BAD_REQUEST,
            details={"start_line": start_line, "total_lines": total_lines},
            start_time=started,
            ctx=ctx,
        )
    if total_lines == 0 and start_line and start_line > 1:
        return error_result(
            message="`start_line` is beyond the end of the file",
            code=ERROR_BAD_REQUEST,
            details={"start_line": start_line, "total_lines": total_lines},
            start_time=started,
            ctx=ctx,
        )

    start, end, pagination_meta = compute_pagination(
        total_lines if total_lines else 0, start_line, line_count
    )

    slice_lines = lines[start - 1 : end] if total_lines else []
    slice_text = "\n".join(slice_lines)
    structured: Dict[str, Any] = {
        "file": {"uri": identity["uri"], "path": identity["relative_path"]},
        "slice": {"start": start, "end": end if total_lines else 0, "total": total_lines},
        "annotated": annotate_lines,
    }

    include_pagination = (
        pagination_meta["has_more"]
        or (start_line is not None and start_line != 1)
        or (line_count is not None)
    )
    if include_pagination:
        structured["pagination"] = pagination_meta

    content_items: List[Dict[str, Any]] = []

    if annotate_lines:
        payload_lines = [
            {"number": start + idx, "text": line}
            for idx, line in enumerate(slice_lines)
        ]
        structured["lines"] = payload_lines
        max_digits = len(str(total_lines if total_lines else 1))
        annotated = "\n".join(
            f"{entry['number']:{max_digits}d}: {entry['text']}"
            for entry in payload_lines
        )
        if annotated and len(annotated) <= 4000:
            content_items.append(_resource_item(identity["uri"], annotated))
    else:
        structured["contents"] = slice_text
        if slice_text and len(slice_text) <= 4000:
            content_items.append(_resource_item(identity["uri"], slice_text))

    if total_lines:
        summary = f"{start}-{end}/{total_lines}"
    else:
        summary = "Empty"

    content_list = [_text_item(summary)]
    content_list.extend(content_items)

    return success_result(
        summary=summary,
        structured=structured,
        start_time=started,
        ctx=ctx,
        content=content_list,
    )


@mcp.tool("lean_diagnostic_messages")
def diagnostic_messages(
    ctx: Context,
    file_path: str,
    start_line: Optional[int] = None,
    line_count: Optional[int] = None,
    _format: Optional[str] = None,
) -> Any:
    """Get all diagnostic msgs (errors, warnings, infos) for a Lean file.

    "no goals to be solved" means code may need removal.

    Args:
        file_path (str): Abs path to Lean file
        start_line (int, optional): 1-based line to start from.
        line_count (int, optional): Number of lines to include.

    Returns:
        List[str] | str: Diagnostic msgs or error msg
    """
    started = time.perf_counter()
    try:
        with open_file_session(ctx, file_path, started=started) as file_session:
            if start_line is not None and start_line < 1:
                return error_result(
                    message="`start_line` must be >= 1",
                    code=ERROR_BAD_REQUEST,
                    details={"start_line": start_line},
                    start_time=started,
                    ctx=ctx,
                )
            if line_count is not None and line_count < 1:
                return error_result(
                    message="`line_count` must be >= 1",
                    code=ERROR_BAD_REQUEST,
                    details={"line_count": line_count},
                    start_time=started,
                    ctx=ctx,
                )

            identity = file_session.identity
            content = file_session.load_content()
            total_lines = len(content.splitlines())
            if total_lines and start_line and start_line > total_lines:
                return error_result(
                    message="`start_line` is beyond the end of the file",
                    code=ERROR_BAD_REQUEST,
                    details={"start_line": start_line, "total_lines": total_lines},
                    start_time=started,
                    ctx=ctx,
                )
            if total_lines == 0 and start_line and start_line > 1:
                return error_result(
                    message="`start_line` is beyond the end of the file",
                    code=ERROR_BAD_REQUEST,
                    details={"start_line": start_line, "total_lines": total_lines},
                    start_time=started,
                    ctx=ctx,
                )

            start, end, pagination_meta = compute_pagination(
                total_lines, start_line, line_count
            )
            start_idx = max(0, start - 1)
            end_idx = max(0, end - 1)

            diagnostics = file_session.client.get_diagnostics(file_session.rel_path)

            if line_count is None and start_line is None:
                filtered = diagnostics
            else:
                filtered = []
                for diag in diagnostics:
                    rng = diag.get("fullRange", diag.get("range"))
                    if not rng:
                        filtered.append(diag)
                        continue
                    diag_start = rng["start"]["line"]
                    diag_end = rng["end"]["line"]
                    if diag_start <= end_idx and diag_end >= start_idx:
                        filtered.append(diag)

            entries = diagnostics_to_entries(filtered)
            summary = summarize_diagnostics(entries)

            include_pagination = (
                pagination_meta["has_more"]
                or (start_line is not None and start_line != 1)
                or (line_count is not None)
            )

            # Build summary line
            error_count = summary["bySeverity"].get("error", 0)
            summary_text = f"{summary['count']} diagnostics ({error_count} errors)."

            # Compact, token-minimizing shape: dedupe messages and compress ranges.
            def _sev_code(entry: Dict[str, Any]) -> Optional[int]:
                code = entry.get("severityCode")
                if isinstance(code, int):
                    return code
                # fallback best-effort from label
                label = entry.get("severity")
                return {"error": 1, "warning": 2, "info": 3, "hint": 4}.get(label)  # type: ignore[arg-type]

            # Unique messages in order of first appearance
            msg_index: Dict[str, int] = {}
            messages: List[str] = []
            diags_compact: List[Dict[str, Any]] = []

            for e in entries:
                msg = e.get("message", "")
                if msg not in msg_index:
                    msg_index[msg] = len(messages)
                    messages.append(msg)
                idx = msg_index[msg]

                rng = e.get("range")
                s_arr = e_arr = None
                if rng and isinstance(rng, dict):
                    try:
                        s_arr = [
                            int(rng["start"]["line"]),
                            int(rng["start"]["character"]),
                        ]
                        e_arr = [
                            int(rng["end"]["line"]),
                            int(rng["end"]["character"]),
                        ]
                    except Exception:
                        s_arr = e_arr = None

                item: Dict[str, Any] = {"m": idx}
                sev = _sev_code(e)
                if sev is not None:
                    item["sev"] = sev
                if s_arr is not None:
                    item["s"] = s_arr
                if e_arr is not None:
                    item["e"] = e_arr
                # include code if available (useful, compact)
                if "code" in e and e["code"] is not None:
                    item["code"] = e["code"]
                diags_compact.append(item)

            # Build numeric bySev map
            by_sev_codes: Dict[str, int] = {}
            for sev_label, count in summary.get("bySeverity", {}).items():
                code = {"error": 1, "warning": 2, "info": 3, "hint": 4}.get(
                    sev_label
                )
                if code is not None:
                    by_sev_codes[str(code)] = count

            structured: Dict[str, Any] = {
                "file": {
                    "uri": identity["uri"],
                    "path": identity["relative_path"],
                },
                "summary": {"count": summary["count"], "bySev": by_sev_codes},
                "messages": messages,
                "diags": diags_compact,
            }
            if include_pagination:
                structured["pagination"] = pagination_meta

            return success_result(
                summary=summary_text,
                structured=structured,
                start_time=started,
                ctx=ctx,
            )
    except ToolError as exc:
        return exc.payload


@mcp.tool("lean_goal")
def goal(
    ctx: Context,
    file_path: str,
    line: int,
    column: Optional[int] = None,
    _format: Optional[str] = None,
) -> Any:
    """Get the proof goals (proof state) at a specific location in a Lean file.

    VERY USEFUL! Main tool to understand the proof state and its evolution!
    Returns "no goals" if solved.
    To see the goal at sorry, use the cursor before the "s".
    Avoid giving a column if unsure-default behavior works well.

    Args:
        file_path (str): Abs path to Lean file
        line (int): Line number (1-indexed)
        column (int, optional): Column number (1-indexed). Defaults to None => Both before and after the line.

    Returns:
        str: Goal(s) or error msg
    """
    started = time.perf_counter()
    try:
        with open_file_session(
            ctx,
            file_path,
            started=started,
            line=line,
            column=column,
        ) as file_session:
            identity = file_session.identity
            client = file_session.client
            rel_path = file_session.rel_path
            content = file_session.load_content()
            lines = content.splitlines()

            if column is None:
                if line < 1 or line > len(lines):
                    message = "Line number out of range. Try elsewhere?"
                    return error_result(
                        message=message,
                        code=ERROR_NOT_GOAL_POSITION,
                        details={"file": identity["relative_path"], "line": line},
                        start_time=started,
                        ctx=ctx,
                    )

                line_text = lines[line - 1]
                column_end = len(line_text)
                column_start = next(
                    (i for i, c in enumerate(line_text) if not c.isspace()), 0
                )
                goal_start = client.get_goal(rel_path, line - 1, column_start)
                goal_end = client.get_goal(rel_path, line - 1, column_end)

                if goal_start is None and goal_end is None:
                    summary_text = f"No goals at line {line}."
                    structured = {
                        "file": {
                            "uri": identity["uri"],
                            "path": identity["relative_path"],
                        },
                        "pos": _compact_pos(line=line),
                        "status": "no_goals",
                        "code": "no_goals",
                        "message": "No goals on that line.",
                    }
                    return success_result(
                        summary=summary_text,
                        structured=structured,
                        content=[_text_item(summary_text)],
                        start_time=started,
                        ctx=ctx,
                    )

                results = []
                if goal_start is not None:
                    results.append(
                        {
                            "kind": "line_start",
                            "position": {"line": line - 1, "character": column_start},
                            "goal": goal_to_payload(goal_start),
                        }
                    )
                if goal_end is not None:
                    results.append(
                        {
                            "kind": "line_end",
                            "position": {"line": line - 1, "character": column_end},
                            "goal": goal_to_payload(goal_end),
                        }
                    )

                primary_goal_text = format_goal(
                    goal_start or goal_end,
                    "No goal text available.",
                )
                summary_text = f"Goals around line {line}."
                content_items = [
                    _text_item(summary_text),
                    _text_item(
                        primary_goal_text.splitlines()[0]
                        if primary_goal_text
                        else ""
                    ),
                ]
                compact_goals: List[Dict[str, Any]] = []
                for r in results:
                    kind = r.get("kind")
                    pos = r.get("position", {})
                    gl = r.get("goal") or {}
                    rendered = (
                        gl.get("rendered") if isinstance(gl, dict) else None
                    )
                    raw_line = pos.get("line")
                    raw_column = pos.get("character")
                    line_index = max((raw_line + 1) if raw_line is not None else 1, 1)
                    column_index = max((raw_column + 1) if raw_column is not None else 1, 1)
                    item: Dict[str, Any] = {
                        "k": ("start" if kind == "line_start" else "end"),
                        "p": [
                            line_index,
                            column_index,
                        ],
                        "r": (
                            rendered.splitlines()[0]
                            if isinstance(rendered, str) and rendered
                            else ""
                        ),
                    }
                    compact_goals.append(item)
                structured = {
                    "file": {
                        "uri": identity["uri"],
                        "path": identity["relative_path"],
                    },
                    "pos": _compact_pos(line=line),
                    "goals": compact_goals,
                }
                return success_result(
                    summary=summary_text,
                    structured=structured,
                    content=content_items,
                    start_time=started,
                    ctx=ctx,
                )

            if column < 1:
                return error_result(
                    message="Column must be >= 1",
                    code=ERROR_BAD_REQUEST,
                    details={"file": identity["relative_path"], "line": line, "column": column},
                    start_time=started,
                    ctx=ctx,
                )

            goal_value = client.get_goal(rel_path, line - 1, column - 1)
            formatted_goal = format_goal(
                goal_value, "Not a valid goal position. Try elsewhere?"
            )
            if goal_value is None:
                summary_text = f"No goals at {line}:{column}."
                structured = {
                    "file": {
                        "uri": identity["uri"],
                        "path": identity["relative_path"],
                    },
                    "pos": _compact_pos(line=line, column=column),
                    "status": "no_goals",
                    "code": "no_goals",
                    "message": "No goals at that position.",
                }
                return success_result(
                    summary=summary_text,
                    structured=structured,
                    content=[_text_item(summary_text)],
                    start_time=started,
                    ctx=ctx,
                )

            summary_text = f"Goal at {line}:{column}."
            content_items = [
                _text_item(summary_text),
                _text_item(
                    formatted_goal.splitlines()[0] if formatted_goal else ""
                ),
            ]
            payload = goal_to_payload(goal_value)
            rendered = payload.get("rendered") if isinstance(payload, dict) else None
            structured = {
                "file": {
                    "uri": identity["uri"],
                    "path": identity["relative_path"],
                },
                "pos": _compact_pos(line=line, column=column),
                "rendered": rendered,
            }
            return success_result(
                summary=summary_text,
                structured=structured,
                content=content_items,
                start_time=started,
                ctx=ctx,
            )
    except ToolError as exc:
        return exc.payload


@mcp.tool("lean_term_goal")
def term_goal(
    ctx: Context, file_path: str, line: int, column: Optional[int] = None, _format: Optional[str] = None
) -> Any:
    """Get the expected type (term goal) at a specific location in a Lean file.

    Args:
        file_path (str): Abs path to Lean file
        line (int): Line number (1-indexed)
        column (int, optional): Column number (1-indexed). Defaults to None => end of line.

    Returns:
        str: Expected type or error msg
    """
    started = time.perf_counter()
    try:
        with open_file_session(
            ctx,
            file_path,
            started=started,
            line=line,
            column=column,
        ) as file_session:
            identity = file_session.identity
            content = file_session.load_content()
            lines = content.splitlines()

            target_column = column
            if target_column is None:
                if line < 1 or line > len(lines):
                    message = "Line number out of range. Try elsewhere?"
                    return error_result(
                        message=message,
                        code=ERROR_NOT_GOAL_POSITION,
                        details={"file": identity["relative_path"], "line": line},
                        start_time=started,
                        ctx=ctx,
                    )
                target_column = len(lines[line - 1])

            if line < 1 or (lines and line > len(lines)):
                return error_result(
                    message="Line number out of range.",
                    code=ERROR_NOT_GOAL_POSITION,
                    details={"file": identity["relative_path"], "line": line},
                    start_time=started,
                    ctx=ctx,
                )

            if target_column is None or target_column < 1:
                return error_result(
                    message="Column must be >= 1",
                    code=ERROR_BAD_REQUEST,
                    details={
                        "file": identity["relative_path"],
                        "line": line,
                        "column": target_column,
                    },
                    start_time=started,
                    ctx=ctx,
                )

            term_goal_value = file_session.client.get_term_goal(
                file_session.rel_path, line - 1, target_column - 1
            )
            if term_goal_value is None:
                return error_result(
                    message="Not a valid term goal position.",
                    code=ERROR_NOT_GOAL_POSITION,
                    details={
                        "file": identity["relative_path"],
                        "line": line,
                        "column": target_column,
                    },
                    start_time=started,
                    ctx=ctx,
                )

            rendered = term_goal_value.get("goal")
            if rendered is not None:
                rendered = rendered.replace("```lean\n", "").replace("\n```", "")

            summary_text = f"Term goal at {line}:{target_column}."
            snippet = rendered.splitlines()[0] if rendered else ""
            content_items = [_text_item(summary_text)]
            if snippet:
                content_items.append(_text_item(snippet))

            structured = {
                "file": {
                    "uri": identity["uri"],
                    "path": identity["relative_path"],
                },
                "pos": _compact_pos(line=line, column=target_column),
                "rendered": rendered,
            }
            return success_result(
                summary=summary_text,
                structured=structured,
                content=content_items,
                start_time=started,
                ctx=ctx,
            )
    except ToolError as exc:
        return exc.payload


@mcp.tool("lean_hover_info")
def hover(ctx: Context, file_path: str, line: int, column: int, _format: Optional[str] = None) -> Any:
    """Get hover info (docs for syntax, variables, functions, etc.) at a specific location in a Lean file.

    Args:
        file_path (str): Abs path to Lean file
        line (int): Line number (1-indexed)
        column (int): Column number (1-indexed). Make sure to use the start or within the term, not the end.

    Returns:
        str: Hover info or error msg
    """
    started = time.perf_counter()
    try:
        with open_file_session(
            ctx,
            file_path,
            started=started,
            line=line,
            column=column,
        ) as file_session:
            identity = file_session.identity
            file_content = file_session.load_content()
            hover_info = file_session.client.get_hover(
                file_session.rel_path, line - 1, column - 1
            )
            if hover_info is None:
                context_line = format_line(file_content, line, column)
                return error_result(
                    message="No hover information at that position.",
                    details={
                        "file": identity["relative_path"],
                        "line": line,
                        "column": column,
                        "context": context_line,
                    },
                    start_time=started,
                    ctx=ctx,
                )

            h_range = hover_info.get("range")
            symbol = extract_range(file_content, h_range)
            info = hover_info["contents"].get(
                "value", "No hover information available."
            )
            info = info.replace("```lean\n", "").replace("\n```", "").strip()

            diagnostics = file_session.client.get_diagnostics(file_session.rel_path)
            filtered = filter_diagnostics_by_position(
                diagnostics, line - 1, column - 1
            )
            diagnostic_entries = diagnostics_to_entries(filtered)

            summary_text = f"Hover for `{symbol}` at {line}:{column}."
            # Compact range and diagnostics
            rng = normalize_range(h_range)
            s_arr = e_arr = None
            if rng:
                try:
                    s_arr = [
                        int(rng["start"]["line"]),
                        int(rng["start"]["character"]),
                    ]
                    e_arr = [
                        int(rng["end"]["line"]),
                        int(rng["end"]["character"]),
                    ]
                except Exception:
                    s_arr = e_arr = None

            # diagnostics compact
            msg_index: Dict[str, int] = {}
            messages: List[str] = []
            diags_compact: List[Dict[str, Any]] = []
            for entry in diagnostic_entries:
                msg = entry.get("message", "")
                if msg not in msg_index:
                    msg_index[msg] = len(messages)
                    messages.append(msg)
                idx = msg_index[msg]
                sev = entry.get("severityCode")
                rng2 = entry.get("range")
                s2 = e2 = None
                if rng2:
                    try:
                        s2 = [
                            int(rng2["start"]["line"]),
                            int(rng2["start"]["character"]),
                        ]
                        e2 = [
                            int(rng2["end"]["line"]),
                            int(rng2["end"]["character"]),
                        ]
                    except Exception:
                        s2 = e2 = None
                item: Dict[str, Any] = {"m": idx}
                if isinstance(sev, int):
                    item["sev"] = sev
                if s2 is not None:
                    item["s"] = s2
                if e2 is not None:
                    item["e"] = e2
                diags_compact.append(item)

            structured = {
                "file": {
                    "uri": identity["uri"],
                    "path": identity["relative_path"],
                },
                "pos": _compact_pos(line=line, column=column),
                "symbol": symbol,
                "range": (
                    {"s": s_arr, "e": e_arr}
                    if s_arr is not None and e_arr is not None
                    else None
                ),
                "infoSnippet": info.splitlines()[0] if info else "",
                "messages": messages,
                "diags": diags_compact,
            }
            content_items = [_text_item(summary_text)]
            if info:
                content_items.append(_text_item(info.splitlines()[0]))

            return success_result(
                summary=summary_text,
                structured=structured,
                content=content_items,
                start_time=started,
                ctx=ctx,
            )
    except ToolError as exc:
        return exc.payload


@mcp.tool("lean_completions")
def completions(
    ctx: Context, file_path: str, line: int, column: int, max_completions: int = 32, _format: Optional[str] = None
) -> Any:
    """Get code completions at a location in a Lean file.

    Only use this on INCOMPLETE lines/statements to check available identifiers and imports:
    - Dot Completion: Displays relevant identifiers after a dot (e.g., `Nat.`, `x.`, or `Nat.ad`).
    - Identifier Completion: Suggests matching identifiers after part of a name.
    - Import Completion: Lists importable files after `import` at the beginning of a file.

    Args:
        file_path (str): Abs path to Lean file
        line (int): Line number (1-indexed)
        column (int): Column number (1-indexed)
        max_completions (int, optional): Maximum number of completions to return. Defaults to 32

    Returns:
        str: List of possible completions or error msg
    """
    started = time.perf_counter()
    try:
        with open_file_session(
            ctx,
            file_path,
            started=started,
            line=line,
            column=column,
        ) as file_session:
            identity = file_session.identity
            content = file_session.load_content()

            completion_items = file_session.client.get_completions(
                file_session.rel_path, line - 1, column - 1
            )
            labels = [item.get("label") for item in completion_items if item.get("label")]

            if not labels:
                return error_result(
                    message="No completions at that position.",
                    details={
                        "file": identity["relative_path"],
                        "line": line,
                        "column": column,
                    },
                    start_time=started,
                    ctx=ctx,
                )

            lines = content.splitlines()
            prefix = ""
            if 0 < line <= len(lines):
                text_before_cursor = (
                    lines[line - 1][: column - 1] if column > 0 else ""
                )
                if not text_before_cursor.endswith("."):
                    prefix = re.split(
                        r"[\s()\[\]{},:;.]+", text_before_cursor
                    )[-1].lower()

            if prefix:

                def sort_key(item: str) -> tuple[int, str]:
                    item_lower = item.lower()
                    if item_lower.startswith(prefix):
                        return (0, item_lower)
                    if prefix in item_lower:
                        return (1, item_lower)
                    return (2, item_lower)

                labels.sort(key=sort_key)
            else:
                labels.sort(key=str.lower)

            truncated = labels[:max_completions]
            if len(labels) > max_completions:
                truncated.append(f"{len(labels) - max_completions} more...")

            suggestions: List[Dict[str, Any]] = []
            for item in completion_items[:max_completions]:
                entry = {"label": item.get("label")}
                for field in ("detail", "kind", "sortText"):
                    if field in item and item[field] is not None:
                        entry[field] = item[field]
                suggestions.append(entry)

            structured = {
                "file": {
                    "uri": identity["uri"],
                    "path": identity["relative_path"],
                },
                "pos": _compact_pos(line=line, column=column),
                "prefix": prefix,
                "labels": [
                    suggestion.get("label")
                    for suggestion in suggestions
                    if suggestion.get("label")
                ],
            }

            summary_text = f"{len(suggestions)} completions at {line}:{column}."
            content_items = [_text_item(summary_text)]
            if truncated:
                content_items.append(_text_item(", ".join(truncated[:3])))

            return success_result(
                summary=summary_text,
                structured=structured,
                content=content_items,
                start_time=started,
                ctx=ctx,
            )
    except ToolError as exc:
        return exc.payload


@mcp.tool("lean_declaration_file")
def declaration_file(ctx: Context, file_path: str, symbol: str, _format: Optional[str] = None) -> Any:
    """Get the file contents where a symbol/lemma/class/structure is declared."""

    started = time.perf_counter()
    try:
        with open_file_session(
            ctx,
            file_path,
            started=started,
            category="lean_declaration_file",
            invalid_details={"symbol": symbol},
            client_details={"symbol": symbol},
        ) as file_session:
            identity = file_session.identity
            orig_file_content = file_session.load_content()

            position = find_start_position(orig_file_content, symbol)
            if not position:
                message = (
                    f"Symbol `{symbol}` not found in `{identity['relative_path']}`."
                )
                return error_result(
                    message=message,
                    code=ERROR_BAD_REQUEST,
                    category="lean_declaration_file",
                    details={
                        "file": identity["relative_path"],
                        "symbol": symbol,
                    },
                    start_time=started,
                    ctx=ctx,
                )

            declaration = file_session.client.get_declarations(
                file_session.rel_path, position["line"], position["column"]
            )
            if not declaration:
                return error_result(
                    message=f"No declaration available for `{symbol}`.",
                    code=ERROR_UNKNOWN,
                    category="lean_declaration_file",
                    details={
                        "file": identity["relative_path"],
                        "symbol": symbol,
                    },
                    start_time=started,
                    ctx=ctx,
                )

            declaration_entry = declaration[0]
            uri = declaration_entry.get("targetUri") or declaration_entry.get("uri")
            abs_path = uri_to_absolute_path(uri)
            if not abs_path or not os.path.exists(abs_path):
                return error_result(
                    message=f"Could not open declaration for `{symbol}`.",
                    code=ERROR_INVALID_PATH,
                    category="lean_declaration_file",
                    details={
                        "symbol": symbol,
                        "path": _sanitize_path_label(abs_path or uri or ""),
                    },
                    start_time=started,
                    ctx=ctx,
                )

            file_content = get_file_contents(abs_path)
            declaration_identity = file_identity(
                _sanitize_path_label(abs_path), absolute_path=abs_path
            )
            structured = {
                "symbol": symbol,
                "origin": {
                    "file": {
                        "uri": identity["uri"],
                        "path": identity["relative_path"],
                    },
                    "pos": _compact_pos(
                        line=position["line"] + 1,
                        column=position["column"] + 1,
                    ),
                },
                "declaration": {
                    "file": {
                        "uri": declaration_identity["uri"],
                        "path": declaration_identity["relative_path"],
                    }
                },
            }

            summary = f"Declaration for `{symbol}`."
            content_items = [_text_item(summary)]
            if file_content and len(file_content) <= 4000:
                content_items.append(
                    _resource_item(declaration_identity["uri"], file_content)
                )

            return success_result(
                summary=summary,
                structured=structured,
                content=content_items,
                start_time=started,
                ctx=ctx,
            )
    except ToolError as exc:
        return exc.payload


@mcp.tool("lean_multi_attempt")
def multi_attempt(
    ctx: Context, file_path: str, line: int, snippets: List[str], _format: Optional[str] = None
) -> Any:
    """Try multiple Lean code snippets at a line and get the goal state and diagnostics for each.

    Use to compare tactics or approaches.
    Use rarely-prefer direct file edits to keep users involved.
    For a single snippet, edit the file and run `lean_diagnostic_messages` instead.

    Note:
        Only single-line, fully-indented snippets are supported.
        Avoid comments for best results.

    Args:
        file_path (str): Abs path to Lean file
        line (int): Line number (1-indexed)
        snippets (List[str]): List of snippets (3+ are recommended)

    Returns:
        List[str] | str: Diagnostics and goal states or error msg
    """

    started = time.perf_counter()
    if not snippets:
        return error_result(
            message="Provide at least one snippet to evaluate.",
            code=ERROR_BAD_REQUEST,
            category="lean_multi_attempt",
            details={"line": line},
            start_time=started,
            ctx=ctx,
        )

    try:
        with open_file_session(
            ctx,
            file_path,
            started=started,
            line=line,
            category="lean_multi_attempt",
        ) as file_session:
            identity = file_session.identity
            client = file_session.client
            rel_path = file_session.rel_path
            file_session.load_content(refresh=True)

            try:
                client.open_file(rel_path)

                _, DocumentChangeCls = ensure_leanclient_available()

                attempts: List[Dict[str, Any]] = []
                snippet_summaries: List[str] = []
                for snippet in snippets:
                    payload = snippet if snippet.endswith("\n") else f"{snippet}\n"
                    change = DocumentChangeCls(
                        payload,
                        [line - 1, 0],
                        [line, 0],
                    )
                    diagnostics_raw = client.update_file(rel_path, [change])
                    diagnostics_entries = diagnostics_to_entries(
                        diagnostics_raw, select_line=line - 1
                    )
                    goal = client.get_goal(rel_path, line - 1, len(snippet))
                    formatted_goal = format_goal(goal, "Missing goal")

                    attempts.append(
                        {
                            "snippet": snippet,
                            "goal": goal_to_payload(goal),
                            "diagnostics": diagnostics_entries,
                        }
                    )

                    diag_summary = summarize_diagnostics(diagnostics_entries)
                    snippet_label = (
                        snippet.strip().splitlines()[0] if snippet.strip() else "<blank>"
                    )
                    snippet_summaries.append(
                        f"`{snippet_label}`: {diag_summary['count']} diagnostics"
                    )

                compact_attempts: List[Dict[str, Any]] = []
                for att in attempts:
                    goal_payload = att.get("goal") or {}
                    rendered_goal = (
                        goal_payload.get("rendered")
                        if isinstance(goal_payload, dict)
                        else None
                    )
                    diag_count = (
                        summarize_diagnostics(att.get("diagnostics", [])).get("count", 0)
                    )
                    compact_attempts.append(
                        {
                            "s": att.get("snippet", ""),
                            "dc": diag_count,
                            "gs": (
                                rendered_goal.splitlines()[0]
                                if isinstance(rendered_goal, str) and rendered_goal
                                else ""
                            ),
                        }
                    )
                structured = {
                    "file": {
                        "uri": identity["uri"],
                        "path": identity["relative_path"],
                    },
                    "pos": _compact_pos(line=line),
                    "attempts": compact_attempts,
                }

                summary = f"Tried {len(attempts)} snippet(s) at line {line}."
                content_items: List[Dict[str, Any]] = [_text_item(summary)]
                if snippet_summaries:
                    content_items.append(
                        _text_item("; ".join(snippet_summaries[:3]))
                    )
                if attempts:
                    first_goal = attempts[0]["goal"]
                    rendered_goal = (
                        first_goal.get("rendered") if isinstance(first_goal, dict) else None
                    )
                    if rendered_goal:
                        content_items.append(_text_item(rendered_goal))

                return success_result(
                    summary=summary,
                    structured=structured,
                    content=content_items,
                    start_time=started,
                    ctx=ctx,
                )
            finally:
                try:
                    client.close_files([rel_path])
                except Exception as exc:  # pragma: no cover - close failures only logged
                    logger.warning(
                        "Failed to close `%s` after multi_attempt: %s", rel_path, exc
                    )
    except ToolError as exc:
        return exc.payload


@mcp.tool("lean_run_code")
def run_code(ctx: Context, code: str, _format: Optional[str] = None) -> Any:
    """Run a complete, self-contained code snippet and return diagnostics.

    Has to include all imports and definitions!
    Only use for testing outside open files! Keep the user in the loop by editing files instead.

    Args:
        code (str): Code snippet

    Returns:
        List[str] | str: Diagnostics msgs or error msg
    """
    started = time.perf_counter()
    lean_project_path = ctx.request_context.lifespan_context.lean_project_path
    if lean_project_path is None:
        message = (
            "No valid Lean project path found. Run another tool (e.g. `lean_diagnostic_messages`) first to set it up or set the LEAN_PROJECT_PATH environment variable."
        )
        return error_result(
            message=message,
            code=ERROR_BAD_REQUEST,
            category="lean_run_code",
            details={"code_length": len(code)},
            start_time=started,
            ctx=ctx,
        )

    rel_path = f"_mcp_snippet_{uuid.uuid4().hex}.lean"
    abs_path = os.path.join(lean_project_path, rel_path)

    try:
        with open(abs_path, "w") as f:
            f.write(code)
    except Exception as e:
        message = f"Error writing code snippet to file `{abs_path}`:\n{str(e)}"
        return error_result(
            message="Error writing code snippet",
            code=ERROR_IO_FAILURE,
            category="lean_run_code",
            details={"path": _sanitize_path_label(abs_path), "error": str(e)},
            start_time=started,
            ctx=ctx,
        )

    # Ensure a Lean client is ready before asking for diagnostics.
    try:
        startup_client(ctx)
    except Exception as e:
        try:
            os.remove(abs_path)
        except Exception:
            pass
        message = f"Error starting Lean client for `{rel_path}`:\n{str(e)}"
        return error_result(
            message="Error starting Lean client",
            code=ERROR_CLIENT_NOT_READY,
            category="lean_run_code",
            details={"path": _sanitize_path_label(abs_path), "error": str(e)},
            start_time=started,
            ctx=ctx,
        )

    diagnostics_payload: List[Dict[str, Any]] | None = None
    formatted_diagnostics: List[str] | None = None
    close_error: str | None = None
    remove_error: str | None = None

    with client_session(ctx) as client:
        if client is None:
            try:
                os.remove(abs_path)
            except Exception:
                pass
            return error_result(
                message=(
                    "Lean client is not available. Run another tool to initialize the project first."
                ),
                code=ERROR_CLIENT_NOT_READY,
                category="lean_run_code",
                details={"path": _sanitize_path_label(abs_path)},
                start_time=started,
                ctx=ctx,
            )

        try:
            client.open_file(rel_path)
            raw_diagnostics = client.get_diagnostics(rel_path)
            diagnostics_payload = diagnostics_to_entries(raw_diagnostics)
            formatted_diagnostics = format_diagnostics(raw_diagnostics)
        finally:
            try:
                client.close_files([rel_path])
            except Exception as exc:  # pragma: no cover - close failures only logged
                close_error = str(exc)
                logger.warning(
                    "Failed to close `%s` after run_code: %s", rel_path, exc
                )
            try:
                os.remove(abs_path)
            except FileNotFoundError:
                pass
            except Exception as e:
                remove_error = str(e)
                logger.warning(
                    "Failed to remove temporary Lean snippet `%s`: %s", abs_path, e
                )

    if remove_error:
        return error_result(
            message=f"Error removing temporary file `{_sanitize_path_label(abs_path)}`",
            code=ERROR_IO_FAILURE,
            category="lean_run_code",
            details={"path": _sanitize_path_label(abs_path), "error": remove_error},
            start_time=started,
            ctx=ctx,
        )
    if close_error:
        return error_result(
            message=f"Error closing temporary Lean document `{rel_path}`",
            code=ERROR_IO_FAILURE,
            category="lean_run_code",
            details={"path": _sanitize_path_label(abs_path), "error": close_error},
            start_time=started,
            ctx=ctx,
        )

    identity = _identity_for_rel_path(ctx, rel_path)
    diagnostics_entries = diagnostics_payload or []
    diag_summary = summarize_diagnostics(diagnostics_entries)
    # compact diagnostics shape
    msg_index: Dict[str, int] = {}
    messages: List[str] = []
    diags_compact: List[Dict[str, Any]] = []
    for e in diagnostics_entries:
        msg = e.get("message", "")
        if msg not in msg_index:
            msg_index[msg] = len(messages)
            messages.append(msg)
        idx = msg_index[msg]
        rng = e.get("range")
        s_arr = e_arr = None
        if rng and isinstance(rng, dict):
            try:
                s_arr = [int(rng["start"]["line"]), int(rng["start"]["character"])]
                e_arr = [int(rng["end"]["line"]), int(rng["end"]["character"])]
            except Exception:
                s_arr = e_arr = None
        item: Dict[str, Any] = {"m": idx}
        sev = e.get("severityCode")
        if isinstance(sev, int):
            item["sev"] = sev
        if s_arr is not None:
            item["s"] = s_arr
        if e_arr is not None:
            item["e"] = e_arr
        if "code" in e and e["code"] is not None:
            item["code"] = e["code"]
        diags_compact.append(item)

    by_sev_codes: Dict[str, int] = {}
    for sev_label, count in diag_summary.get("bySeverity", {}).items():
        code = {"error": 1, "warning": 2, "info": 3, "hint": 4}.get(sev_label)
        if code is not None:
            by_sev_codes[str(code)] = count

    structured = {
        "file": {"uri": identity["uri"], "path": identity["relative_path"]},
        "summary": {"count": diag_summary["count"], "bySev": by_sev_codes},
        "messages": messages,
        "diags": diags_compact,
    }

    summary_text = (
        "No diagnostics for snippet." if not diag_summary["count"] else (
            f"{diag_summary['count']} diagnostic(s) for snippet."
        )
    )
    content_items = [_text_item(summary_text)]
    if formatted_diagnostics:
        joined = "\n\n".join(formatted_diagnostics)
        if joined and len(joined) <= 4000:
            content_items.append(_resource_item(identity["uri"], joined))

    return success_result(
        summary=summary_text,
        structured=structured,
        content=content_items,
        start_time=started,
        ctx=ctx,
    )


@mcp.tool("lean_tool_spec")
def tool_spec(ctx: Context, _format: Optional[str] = None) -> Any:
    started = time.perf_counter()
    spec = build_tool_spec()
    summary = "Lean MCP tool specification ready."
    tool_names = [t.get("name") for t in spec.get("tools", [])]
    response_kinds = list((spec.get("responses") or {}).keys())
    compact = {
        "tools": tool_names,
        "responses": response_kinds,
    }
    return success_result(
        summary=summary,
        structured=compact,
        content=[_text_item(summary)],
        start_time=started,
        ctx=ctx,
    )


@mcp.tool("lean_leansearch")
@rate_limited("leansearch", max_requests=3, per_seconds=30)
def leansearch(ctx: Context, query: str, num_results: int = 5, _format: Optional[str] = None) -> Any:
    """Search for Lean theorems, definitions, and tactics using leansearch.net.

    Query patterns:
      - Natural language: "If there exist injective maps of sets from A to B and from B to A, then there exists a bijective map between A and B."
      - Mixed natural/Lean: "natural numbers. from: n < m, to: n + 1 < m + 1", "n + 1 <= m if n < m"
      - Concept names: "Cauchy Schwarz"
      - Lean identifiers: "List.sum", "Finset induction"
      - Lean term: "{f : A → B} {g : B → A} (hf : Injective f) (hg : Injective g) : ∃ h, Bijective h"

    Args:
        query (str): Search query
        num_results (int, optional): Max results. Defaults to 5.

    Returns:
        List[Dict] | str: Search results or error msg
    """
    started = time.perf_counter()
    try:
        headers = {"User-Agent": "lean-lsp-mcp/0.1", "Content-Type": "application/json"}
        payload = json.dumps(
            {"num_results": str(num_results), "query": [query]}
        ).encode("utf-8")

        req = urllib.request.Request(
            "https://leansearch.net/search",
            data=payload,
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=20) as response:
            results = json.loads(response.read().decode("utf-8"))

        if not results or not results[0]:
            return error_result(
                message="No results found.",
                code=ERROR_UNKNOWN,
                category="lean_leansearch",
                details={"query": query, "num_results": num_results},
                start_time=started,
                ctx=ctx,
            )
        results = results[0][:num_results]
        results = [r["result"] for r in results]

        for result in results:
            result.pop("docstring", None)

            module_parts = result.get("module_name")
            if isinstance(module_parts, list):
                result["module_name"] = ".".join(module_parts)

            name_parts = result.get("name")
            if isinstance(name_parts, list):
                result["name"] = ".".join(name_parts)

        names = [str(res.get("name") or res.get("declaration") or "") for res in results]
        structured = {"query": query, "names": names}
        preview = ", ".join(filter(None, names[:3]))
        summary = f"{len(results)} results"
        return success_result(
            summary=summary,
            structured=structured,
            start_time=started,
            ctx=ctx,
        )
    except Exception as e:
        return error_result(
            message="leansearch error",
            code=ERROR_UNKNOWN,
            category="lean_leansearch",
            details={"error": str(e), "query": query},
            start_time=started,
            ctx=ctx,
        )


@mcp.tool("lean_loogle")
@rate_limited("loogle", max_requests=3, per_seconds=30)
def loogle(ctx: Context, query: str, num_results: int = 8, _format: Optional[str] = None) -> Any:
    """Search for definitions and theorems using loogle.

    Query patterns:
      - By constant: Real.sin  # finds lemmas mentioning Real.sin
      - By lemma name: "differ"  # finds lemmas with "differ" in the name
      - By subexpression: _ * (_ ^ _)  # finds lemmas with a product and power
      - Non-linear: Real.sqrt ?a * Real.sqrt ?a
      - By type shape: (?a -> ?b) -> List ?a -> List ?b
      - By conclusion: |- tsum _ = _ * tsum _
      - By conclusion w/hyps: |- _ < _ → tsum _ < tsum _

    Args:
        query (str): Search query
        num_results (int, optional): Max results. Defaults to 8.

    Returns:
        List[dict] | str: Search results or error msg
    """
    started = time.perf_counter()
    try:
        req = urllib.request.Request(
            f"https://loogle.lean-lang.org/json?q={urllib.parse.quote(query)}",
            headers={"User-Agent": "lean-lsp-mcp/0.1"},
            method="GET",
        )

        with urllib.request.urlopen(req, timeout=20) as response:
            results = json.loads(response.read().decode("utf-8"))

        if "hits" not in results:
            return error_result(
                message="No results found.",
                code=ERROR_UNKNOWN,
                category="lean_loogle",
                details={"query": query, "num_results": num_results},
                start_time=started,
                ctx=ctx,
            )

        results = results["hits"][:num_results]
        if not results:
            return error_result(
                message="No results found.",
                code=ERROR_UNKNOWN,
                category="lean_loogle",
                details={"query": query, "num_results": num_results},
                start_time=started,
                ctx=ctx,
            )
        for result in results:
            result.pop("doc", None)
        names = [hit.get("name") for hit in results if hit.get("name")]
        structured = {"query": query, "names": names}
        preview = ", ".join(names[:3])
        summary = f"{len(results)} results"
        return success_result(
            summary=summary,
            structured=structured,
            start_time=started,
            ctx=ctx,
        )
    except Exception as e:
        return error_result(
            message="loogle error",
            code=ERROR_UNKNOWN,
            category="lean_loogle",
            details={"error": str(e), "query": query},
            start_time=started,
            ctx=ctx,
        )


@mcp.tool("lean_state_search")
@rate_limited("lean_state_search", max_requests=3, per_seconds=30)
def state_search(
    ctx: Context, file_path: str, line: int, column: int, num_results: int = 5, _format: Optional[str] = None
) -> Any:
    """Search for theorems based on proof state using premise-search.com.

    Only uses first goal if multiple.

    Args:
        file_path (str): Abs path to Lean file
        line (int): Line number (1-indexed)
        column (int): Column number (1-indexed)
        num_results (int, optional): Max results. Defaults to 5.

    Returns:
        List | str: Search results or error msg
    """
    started = time.perf_counter()
    try:
        with open_file_session(
            ctx,
            file_path,
            started=started,
            line=line,
            column=column,
            category="lean_state_search",
        ) as file_session:
            identity = file_session.identity
            file_contents = file_session.load_content()
            goal_state = file_session.client.get_goal(
                file_session.rel_path, line - 1, column - 1
            )
    except ToolError as exc:
        return exc.payload

    if not goal_state or not goal_state.get("goals"):
        return error_result(
            message="No goals found",
            code=ERROR_NO_GOAL,
            category="lean_state_search",
            details={
                "file": identity["relative_path"],
                "line": line,
                "column": column,
            },
            start_time=started,
            ctx=ctx,
        )

    data = {
        "state": goal_state["goals"][0],
        "limit": num_results,
        "query": f_line,
    }

    try:
        headers = {"User-Agent": "lean-lsp-mcp/0.1", "Content-Type": "application/json"}
        req = urllib.request.Request(
            "https://premise-search.com/api/search",
            headers=headers,
            method="POST",
            data=json.dumps(data).encode("utf-8"),
        )

        with urllib.request.urlopen(req, timeout=20) as response:
            results = json.loads(response.read().decode("utf-8"))

        if not results:
            return error_result(
                message="No results found",
                code=ERROR_UNKNOWN,
                category="lean_state_search",
                details={
                    "file": identity["relative_path"],
                    "line": line,
                    "column": column,
                    "limit": num_results,
                },
                start_time=started,
                ctx=ctx,
            )

        names = [res.get("name") or "" for res in results]
        structured = {
            "file": {
                "uri": identity["uri"],
                "path": identity["relative_path"],
            },
            "pos": _compact_pos(line=line, column=column),
            "query": {"state": data["state"], "limit": num_results},
            "names": names,
        }
        summary = f"{len(results)} results"
        return success_result(
            summary=summary,
            structured=structured,
            start_time=started,
            ctx=ctx,
        )
    except Exception as e:
        return error_result(
            message="state search error",
            code=ERROR_UNKNOWN,
            category="lean_state_search",
            details={
                "error": str(e),
                "line": line,
                "column": column,
                "file": identity["relative_path"],
            },
            start_time=started,
            ctx=ctx,
        )


@mcp.tool("lean_hammer_premise")
@rate_limited("hammer_premise", max_requests=3, per_seconds=30)
def hammer_premise(
    ctx: Context, file_path: str, line: int, column: int, num_results: int = 32, _format: Optional[str] = None
) -> Any:
    """Search for premises based on proof state using the lean hammer premise search.

    Args:
        file_path (str): Abs path to Lean file
        line (int): Line number (1-indexed)
        column (int): Column number (1-indexed)
        num_results (int, optional): Max results. Defaults to 32.

    Returns:
        List[str] | str: List of relevant premises or error message
    """
    started = time.perf_counter()
    try:
        with open_file_session(
            ctx,
            file_path,
            started=started,
            line=line,
            column=column,
            category="lean_hammer_premise",
        ) as file_session:
            identity = file_session.identity
            file_contents = file_session.load_content()
            goal_state = file_session.client.get_goal(
                file_session.rel_path, line - 1, column - 1
            )
    except ToolError as exc:
        return exc.payload

    f_line = format_line(file_contents, line, column)
    if not goal_state or not goal_state.get("goals"):
        return error_result(
            message="No goals found",
            code=ERROR_NO_GOAL,
            category="lean_hammer_premise",
            details={
                "file": identity["relative_path"],
                "line": line,
                "column": column,
            },
            start_time=started,
            ctx=ctx,
        )

    data = {
        "state": goal_state["goals"][0],
        "new_premises": [],
        "k": num_results,
    }

    try:
        url = os.getenv("LEAN_HAMMER_URL", "http://leanpremise.net")
        req = urllib.request.Request(
            url + "/retrieve",
            headers={
                "User-Agent": "lean-lsp-mcp/0.1",
                "Content-Type": "application/json",
            },
            method="POST",
            data=json.dumps(data).encode("utf-8"),
        )

        with urllib.request.urlopen(req, timeout=20) as response:
            results = json.loads(response.read().decode("utf-8"))

        results = [result["name"] for result in results]
        structured = {
            "file": {
                "uri": identity["uri"],
                "path": identity["relative_path"],
            },
            "pos": _compact_pos(line=line, column=column),
            "query": {"state": data["state"], "limit": num_results},
            "names": results,
        }
        summary = f"{len(results)} results"
        return success_result(
            summary=summary,
            structured=structured,
            start_time=started,
            ctx=ctx,
        )
    except Exception as e:
        return error_result(
            message="lean hammer premise error",
            code=ERROR_UNKNOWN,
            category="lean_hammer_premise",
            details={
                "error": str(e),
                "file": identity["relative_path"],
                "line": line,
                "column": column,
            },
            start_time=started,
            ctx=ctx,
        )


if __name__ == "__main__":
    mcp.run()
