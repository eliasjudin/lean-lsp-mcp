from __future__ import annotations

import functools
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, TypeVar, cast

from mcp.server.fastmcp import Context

from lean_lsp_mcp.file_utils import update_file
from lean_lsp_mcp.leanclient_provider import LeanclientNotInstalledError
from lean_lsp_mcp.response_formatter import (
    JSON_RESPONSE_FORMAT,
    apply_character_limit,
    build_markdown_summary,
    extend_structured_with_truncation,
    normalize_response_format,
)
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
from lean_lsp_mcp.utils import file_identity

from .context import client_session


class ToolError(Exception):
    def __init__(self, payload: Dict[str, Any]):
        super().__init__(payload.get("structured", {}).get("message", "Tool error"))
        self.payload = payload


_SAFE_EXCEPTION_MESSAGES: Dict[str, str] = {
    "LeanclientNotInstalledError": (
        "`leanclient` is not installed. Install it with `pip install leanclient` to enable Lean LSP integration."
    ),
    "OSError": "File system operation failed.",
    "IOError": "File system operation failed.",
    "FileNotFoundError": "File not found.",
    "PermissionError": "Permission denied.",
    "TimeoutError": "Operation timed out.",
    "ConnectionError": "Connection failed.",
    "URLError": "Network request failed.",
    "HTTPError": "HTTP request failed.",
    "JSONDecodeError": "Invalid JSON response.",
    "UnicodeDecodeError": "Text encoding error.",
    "ValueError": "Invalid value provided.",
    "TypeError": "Invalid type provided.",
}


def sanitize_exception(exc: BaseException, *, fallback_reason: str | None = None) -> str:
    """Sanitize exception messages to avoid exposing internal details.
    
    This function ensures that exception messages shown to users never contain:
    - Internal file paths or system information
    - Stack traces or debugging details
    - Sensitive configuration data
    - Implementation-specific error codes
    
    Args:
        exc: The exception to sanitize
        fallback_reason: Optional context about what operation failed
    
    Returns:
        A safe, user-friendly error message that guides the user toward resolution
    """
    name = exc.__class__.__name__ or "Exception"
    
    # Check for exact class name match first
    safe_message = _SAFE_EXCEPTION_MESSAGES.get(name)
    if safe_message:
        return safe_message
    
    # Check for common base classes with more specific handling
    import urllib.error
    if isinstance(exc, FileNotFoundError):
        return "File not found. Verify the file path is correct."
    if isinstance(exc, PermissionError):
        return "Permission denied. Check file or directory permissions."
    if isinstance(exc, TimeoutError):
        return "Operation timed out. Try again or check network connectivity."
    if isinstance(exc, (OSError, IOError)):
        return "File system operation failed. Check file paths and permissions."
    if isinstance(exc, urllib.error.HTTPError):
        # Don't expose HTTP status codes or URLs
        return "HTTP request failed. Check network connectivity and API availability."
    if isinstance(exc, urllib.error.URLError):
        return "Network request failed. Verify network connectivity."
    if isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError)):
        return "Response parsing failed. The service may have returned invalid data."
    if isinstance(exc, (ValueError, TypeError)):
        return "Invalid input provided. Check parameter types and values."
    if isinstance(exc, ConnectionError):
        return "Connection failed. Check network connectivity."
    
    # Generic fallback with context - never expose exception details
    if fallback_reason:
        return f"Operation failed: {fallback_reason}. Please try again."
    return "An unexpected error occurred. Please try again."


class LeanFileSession:
    def __init__(
        self,
        *,
        ctx: Context,
        client: Any,
        rel_path: str,
        identity: FileIdentity,
        content: str | None = None,
    ) -> None:
        self.ctx = ctx
        self.client = client
        self.rel_path = rel_path
        self.identity = identity
        self._content = content

    def load_content(self, *, refresh: bool = False) -> str:
        if refresh or self._content is None:
            self._content = update_file(self.ctx, self.rel_path)
        return self._content

    def clear_cache(self) -> None:
        self._content = None


def _set_response_format_hint(ctx: Context | None, response_format: Optional[str]) -> None:
    if ctx is None:
        return
    request_context = getattr(ctx, "request_context", None)
    if request_context is not None:
        setattr(request_context, "_response_format_hint", response_format)


def _text_item(text: str) -> Dict[str, str]:
    return {"type": "text", "text": text}


def _resource_item(uri: str, text: str, mime_type: str = "text/plain") -> Dict[str, Any]:
    return {
        "type": "resource",
        "resource": {"uri": uri, "mimeType": mime_type, "text": text},
    }


def _json_item(structured: Dict[str, Any]) -> Dict[str, Any]:
    try:
        json_text = json.dumps(structured, ensure_ascii=False)
    except (TypeError, ValueError):
        json_text = "{}"
    return _resource_item(
        f"file:///_mcp_structured_{uuid.uuid4().hex}.json",
        json_text,
        mime_type="application/json",
    )


def success_result(
    *,
    summary: str,
    structured: Dict[str, Any] | None,
    start_time: float,
    ctx: Context | None = None,
    content: List[Dict[str, Any]] | None = None,
    response_format: str | None = None,
) -> Dict[str, Any]:
    additional_items = list(content or [])
    if additional_items:
        first = additional_items[0]
        if first.get("type") == "text" and first.get("text") == summary:
            additional_items = additional_items[1:]

    if response_format is None and ctx is not None:
        request_context = getattr(ctx, "request_context", None)
        if request_context is not None:
            response_format = getattr(request_context, "_response_format_hint", None)

    selected_format = normalize_response_format(response_format)
    summary_markdown = build_markdown_summary(summary)
    human_items = [_text_item(summary_markdown), *additional_items]

    limited_items, truncated, truncated_sections = apply_character_limit(human_items)
    structured_payload = extend_structured_with_truncation(
        structured,
        truncated=truncated,
        truncated_sections=truncated_sections,
    )

    summary_item: Dict[str, Any] | None = None
    remaining_items: List[Dict[str, Any]] = []
    for item in limited_items:
        if summary_item is None and item.get("type") == "text":
            summary_item = item
            continue
        remaining_items.append(item)

    payload: List[Dict[str, Any]]
    if selected_format == JSON_RESPONSE_FORMAT:
        if structured_payload is not None:
            meta = structured_payload.setdefault("_meta", {})
            if summary:
                meta.setdefault("summary", summary)
            structured_for_json = structured_payload
        else:
            structured_for_json = {"summary": summary} if summary else {}

        payload = []
        payload.append(_json_item(structured_for_json))
        payload.extend(remaining_items)

        return mcp_result(
            content=payload or [_text_item(summary_markdown)],
            structured=structured_for_json if structured_for_json else None,
        )

    payload = []
    if summary_item is not None:
        payload.append(summary_item)
    else:
        payload.append(_text_item(summary_markdown))
    payload.extend(remaining_items)

    return mcp_result(
        content=payload,
        structured=structured_payload,
    )


def _derive_error_hints(
    *,
    code: str | None,
    category: str | None,
    details: Dict[str, Any] | None,
    ctx: Context | None,
) -> List[str]:
    hints: List[str] = []

    def _append(text: str | None) -> None:
        candidate = (text or "").strip()
        if candidate and candidate not in hints:
            hints.append(candidate)

    request_context = None
    lifespan = None
    if ctx is not None:
        request_context = getattr(ctx, "request_context", None)
        if request_context is not None:
            lifespan = getattr(request_context, "lifespan_context", None)
    project_root = getattr(lifespan, "lean_project_path", None) if lifespan else None

    if code == ERROR_CLIENT_NOT_READY:
        _append("Run `lean_build` to initialize the Lean project and restart the Lean LSP client.")

    if not project_root:
        _append(
            "Set `LEAN_PROJECT_PATH` or pass `lean_project_path` to the tool so the server knows your project root."
        )

    if code == ERROR_BAD_REQUEST:
        _append("Call `lean_tool_spec` to review required parameters and defaults for this tool.")

    if code == ERROR_IO_FAILURE:
        detail_keys = details.keys() if isinstance(details, dict) else []
        if "output" in detail_keys or "error" in detail_keys:
            _append("Inspect the failure details (e.g. `output` or `error`) to fix the underlying Lean command before retrying.")
        _append("Run `lean_build` after addressing the issue to verify the project compiles cleanly.")

    if code == ERROR_RATE_LIMIT:
        _append("Wait a few seconds before retrying, or reduce duplicate requests to stay within the rate limit.")

    if code == ERROR_NO_GOAL:
        _append("Ensure the requested location has an active Lean goal (for example, inside an unfinished proof).")

    if code == ERROR_NOT_GOAL_POSITION:
        _append("Move the cursor to a goal position or adjust the Lean code so a goal is produced at that location before retrying.")

    if not hints:
        _append("If the problem persists, rerun `lean_build` to refresh the Lean client and try again.")

    return hints


def error_result(
    *,
    message: str,
    start_time: float,
    ctx: Context | None = None,
    code: str | None = None,
    category: str | None = None,
    details: Dict[str, Any] | None = None,
    hints: List[str] | None = None,
    response_format: str | None = None,
) -> Dict[str, Any]:
    structured: Dict[str, Any] = {"message": message}
    if code:
        structured["code"] = code
    if category:
        structured["category"] = category
    if details:
        structured["details"] = details

    combined_hints: List[str] = []
    derived_hints = _derive_error_hints(
        code=code,
        category=category,
        details=details,
        ctx=ctx,
    )
    for source in (hints or [], derived_hints):
        for hint in source:
            if hint not in combined_hints:
                combined_hints.append(hint)
    if combined_hints:
        structured["hints"] = combined_hints

    detail_bullets: List[str] = []
    if code:
        detail_bullets.append(f"Code: `{code}`")
    if category:
        detail_bullets.append(f"Category: {category}")
    if combined_hints:
        detail_bullets.extend(f"Hint: {hint}" for hint in combined_hints)

    trimmed_message = message.strip()
    if trimmed_message:
        if trimmed_message.casefold().startswith("error"):
            summary_headline = trimmed_message
        else:
            summary_headline = f"Error: {trimmed_message}"
    else:
        summary_headline = "Error"

    if response_format is None and ctx is not None:
        request_context = getattr(ctx, "request_context", None)
        if request_context is not None:
            response_format = getattr(request_context, "_response_format_hint", None)

    selected_format = normalize_response_format(response_format)
    summary_markdown = build_markdown_summary(summary_headline, detail_bullets)
    limited_items, truncated, truncated_sections = apply_character_limit(
        [_text_item(summary_markdown)]
    )
    structured_payload = extend_structured_with_truncation(
        structured,
        truncated=truncated,
        truncated_sections=truncated_sections,
    )

    summary_item: Dict[str, Any] | None = None
    other_items: List[Dict[str, Any]] = []
    for item in limited_items:
        if summary_item is None and item.get("type") == "text":
            summary_item = item
            continue
        other_items.append(item)

    payload: List[Dict[str, Any]]
    if selected_format == JSON_RESPONSE_FORMAT:
        structured_for_json: Dict[str, Any] = (
            structured_payload if structured_payload is not None else {}
        )
        meta = structured_for_json.setdefault("_meta", {})
        meta.setdefault("summary", summary_headline)
        payload = []
        payload.append(_json_item(structured_for_json))
        payload.extend(other_items)
        return mcp_result(
            content=payload,
            structured=structured_for_json if structured_for_json else None,
            is_error=True,
        )

    payload = []
    if summary_item is not None:
        payload.append(summary_item)
    else:
        payload.append(_text_item(summary_markdown))
    payload.extend(other_items)

    return mcp_result(
        content=payload,
        structured=structured_payload,
        is_error=True,
    )


def _sanitize_path_label(path: str) -> str:
    if not path:
        return path
    try:
        path_obj = Path(path).expanduser()
    except TypeError:
        return ""
    try:
        resolved = path_obj.resolve(strict=False)
    except OSError:
        if path_obj.is_absolute():
            resolved = path_obj
        else:
            resolved = (Path.cwd() / path_obj).resolve(strict=False)
    except RuntimeError:
        resolved = path_obj.absolute()
    allowed_roots: List[Path] = [Path.cwd()]
    env_root = os.environ.get("LEAN_PROJECT_PATH")
    if env_root:
        try:
            allowed_roots.append(Path(env_root).expanduser().resolve(strict=False))
        except OSError:
            pass
        except RuntimeError:
            allowed_roots.append(Path(env_root).expanduser().absolute())
    for root in allowed_roots:
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        sanitized = relative.as_posix()
        if sanitized:
            return sanitized
        return "."
    fallback = resolved.name or path_obj.name
    if fallback:
        return fallback
    parts = resolved.parts
    if parts:
        return parts[-1]
    return path_obj.as_posix()


sanitize_path_label = _sanitize_path_label


_SERVER_MODULE_VAR: ContextVar[Any | None] = ContextVar(
    "_SERVER_MODULE_VAR", default=None
)


def register_server_module(module: Any | None) -> None:
    _SERVER_MODULE_VAR.set(module)


@functools.lru_cache(maxsize=1)
def _get_setup_client_for_file() -> Callable[[Context, str], str | None]:
    from lean_lsp_mcp.client_utils import setup_client_for_file

    return setup_client_for_file


def _setup_client_for_file_fallback(ctx: Context, file_path: str) -> str | None:
    return _get_setup_client_for_file()(ctx, file_path)


def _resolve_server_attr(name: str, fallback: Any) -> Any:
    server_module = _SERVER_MODULE_VAR.get()
    if server_module is not None:
        try:
            # Allow either a module object or a globals() dict
            if isinstance(server_module, dict):
                attr = server_module.get(name, None)
            else:
                attr = getattr(server_module, name, None)
        except (AttributeError, TypeError):
            attr = None
        if attr is not None:
            return attr
    server_module = sys.modules.get("lean_lsp_mcp.server")
    if server_module is not None:
        attr = getattr(server_module, name, None)
        if attr is not None:
            return attr
    return fallback


def _identity_for_rel_path(ctx: Context, rel_path: str) -> FileIdentity:
    project_root = ctx.request_context.lifespan_context.lean_project_path
    absolute = None
    if project_root:
        absolute = os.path.join(project_root, rel_path)
    elif os.path.isabs(rel_path):
        absolute = rel_path
    return file_identity(rel_path, absolute)


def _compact_pos(*, line: int | None = None, column: int | None = None) -> Dict[str, int]:
    pos: Dict[str, int] = {}
    if line is not None:
        pos["l"] = max(line, 1)
    if column is not None:
        pos["c"] = max(column, 1)
    return pos


def _normalize_range_to_arrays(rng: Mapping[str, Any] | None) -> tuple[List[int] | None, List[int] | None]:
    """Normalize an LSP range to start and end arrays.
    
    Args:
        rng: LSP range dict with start/end positions
        
    Returns:
        Tuple of (start_array, end_array) or (None, None) if invalid
    """
    if not rng:
        return None, None
    try:
        start = [
            int(rng["start"]["line"]),
            int(rng["start"]["character"]),
        ]
        end = [
            int(rng["end"]["line"]),
            int(rng["end"]["character"]),
        ]
        return start, end
    except (KeyError, TypeError, ValueError):
        return None, None


def _compact_diagnostics(
    diagnostic_entries: Sequence[Mapping[str, Any]],
) -> tuple[List[str], List[Dict[str, Any]]]:
    """Compact diagnostic entries with message deduplication.
    
    Args:
        diagnostic_entries: List of diagnostic entry dicts with message, severityCode, range, etc.
        
    Returns:
        Tuple of (messages list, compact diags list) where compact diags reference messages by index
    """
    msg_index: Dict[str, int] = {}
    messages: List[str] = []
    diags_compact: List[Dict[str, Any]] = []
    
    for entry in diagnostic_entries:
        msg = entry.get("message", "")
        if msg not in msg_index:
            msg_index[msg] = len(messages)
            messages.append(msg)
        idx = msg_index[msg]
        
        rng = entry.get("range")
        s_arr, e_arr = _normalize_range_to_arrays(rng)
        
        item: Dict[str, Any] = {"m": idx}
        sev = entry.get("severityCode")
        if isinstance(sev, int):
            item["sev"] = sev
        if s_arr is not None:
            item["s"] = s_arr
        if e_arr is not None:
            item["e"] = e_arr
        if "code" in entry and entry["code"] is not None:
            item["code"] = entry["code"]
        diags_compact.append(item)
    
    return messages, diags_compact


def retrieve_goal_state_with_session(
    file_session: LeanFileSession,
    line: int,
    column: int
) -> Dict[str, Any] | None:
    """Retrieve goal state from a file session.
    
    Args:
        file_session: Open file session with client access
        line: 1-based line number
        column: 1-based column number
        
    Returns:
        Goal state dict or None if no goal at position
    """
    result = file_session.client.get_goal(
        file_session.rel_path,
        line - 1,
        column - 1,
    )
    return cast(Optional[Dict[str, Any]], result)


@contextmanager
def open_file_session(
    ctx: Context,
    file_path: str,
    *,
    started: float,
    response_format: Optional[str] = None,
    line: int | None = None,
    column: int | None = None,
    category: str | None = None,
    invalid_details: Dict[str, Any] | None = None,
    client_details: Dict[str, Any] | None = None,
    invalid_message: str = "Invalid Lean file path: Unable to start LSP server or load file",
    client_message: str = (
        "Lean client is not available. Run another tool to initialize the project first."
    ),
    setup_override: Any | None = None,
    identity_override: Any | None = None,
) -> Iterator[LeanFileSession]:
    invalid_payload = dict(invalid_details or {})
    invalid_payload.setdefault("file_path", sanitize_path_label(file_path))
    if line is not None:
        invalid_payload.setdefault("line", line)
    if column is not None:
        invalid_payload.setdefault("column", column)

    setup_fn = (
        setup_override
        if setup_override is not None
        else _resolve_server_attr("setup_client_for_file", _setup_client_for_file_fallback)
    )
    identity_for_rel_path_fn = (
        identity_override
        if identity_override is not None
        else _resolve_server_attr("_identity_for_rel_path", _identity_for_rel_path)
    )

    try:
        rel_path = setup_fn(ctx, file_path)
    except LeanclientNotInstalledError as exc:
        dependency_payload = dict(invalid_payload)
        dependency_payload.setdefault("dependency", "leanclient")
        raise ToolError(
            error_result(
                message=sanitize_exception(exc),
                code=ERROR_CLIENT_NOT_READY,
                category=category,
                details=dependency_payload,
                start_time=started,
                ctx=ctx,
                response_format=response_format,
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
                response_format=response_format,
            )
        )

    identity = identity_for_rel_path_fn(ctx, rel_path)
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
                    response_format=response_format,
                )
            )

        yield LeanFileSession(
            ctx=ctx,
            client=client,
            rel_path=rel_path,
            identity=identity,
        )


F = TypeVar("F", bound=Callable[..., Any])


def rate_limited(category: str, max_requests: int, per_seconds: int) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        docstring = func.__doc__ or ""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            ctx = kwargs.get("ctx")
            if ctx is None:
                if args:
                    ctx = args[0]
                else:
                    raise TypeError("rate-limited tool requires a Context as the first argument")

            request_context = getattr(ctx, "request_context", None)
            if request_context is None:
                return error_result(
                    message="Missing request context for rate-limited tool",
                    code=ERROR_UNKNOWN,
                    category=category,
                    details={"reason": "request_context not set"},
                    start_time=started,
                    ctx=ctx,
                )

            lifespan = getattr(request_context, "lifespan_context", None)
            if lifespan is None:
                return error_result(
                    message="Missing lifespan context for rate-limited tool",
                    code=ERROR_UNKNOWN,
                    category=category,
                    details={"reason": "lifespan_context not set"},
                    start_time=started,
                    ctx=ctx,
                )
            lock = getattr(lifespan, "rate_limit_lock", None)
            lock_ctx = lock if lock is not None else nullcontext()

            with lock_ctx:
                rate_limit = getattr(lifespan, "rate_limit", None)
                if rate_limit is None:
                    rate_limit = {}
                    setattr(lifespan, "rate_limit", rate_limit)

                timestamps = rate_limit.setdefault(category, [])
                current_time = time.monotonic()
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

        wrapper_globals = cast(Dict[str, Any], wrapper.__globals__)
        for name, value in func.__globals__.items():
            wrapper_globals.setdefault(name, value)

        limit_prefix = f"Rate limit: {max_requests} requests every {per_seconds} seconds."
        wrapper.__doc__ = f"{limit_prefix} {docstring}" if docstring else limit_prefix
        return cast(F, wrapper)

    return decorator


__all__ = [
    "ToolError",
    "LeanFileSession",
    "_set_response_format_hint",
    "_text_item",
    "_resource_item",
    "_json_item",
    "success_result",
    "_derive_error_hints",
    "error_result",
    "_sanitize_path_label",
    "sanitize_exception",
    "_identity_for_rel_path",
    "_compact_pos",
    "_normalize_range_to_arrays",
    "_compact_diagnostics",
    "retrieve_goal_state_with_session",
    "open_file_session",
    "client_session",
    "rate_limited",
    "register_server_module",
]
