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
import inspect
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
from lean_lsp_mcp.tool_spec import TOOL_ANNOTATIONS, TOOL_SPEC_VERSION, build_tool_spec
from lean_lsp_mcp.schema import mcp_result
from lean_lsp_mcp.response_formatter import (
    JSON_RESPONSE_FORMAT,
    apply_character_limit,
    build_markdown_summary,
    extend_structured_with_truncation,
    normalize_response_format,
)
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
from lean_lsp_mcp.tool_inputs import (
    LeanBuildInput,
    LeanCompletionsInput,
    LeanDeclarationFileInput,
    LeanDiagnosticMessagesInput,
    LeanFileContentsInput,
    LeanGoalInput,
    LeanHammerPremiseInput,
    LeanHoverInput,
    LeanMultiAttemptInput,
    LeanRunCodeInput,
    LeanSearchInput,
    LeanStateSearchInput,
    LeanTermGoalInput,
    LeanToolSpecInput,
    LoogleSearchInput,
)
from lean_lsp_mcp.leanclient_provider import (
    LeanclientNotInstalledError,
    ensure_leanclient_available,
    is_leanclient_available,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for typing only
    from leanclient import DocumentContentChange, LeanLSPClient
else:
    DocumentContentChange = Any  # type: ignore[assignment]
    LeanLSPClient = Any  # type: ignore[assignment]


try:  # pragma: no cover - metadata lookup may fail in tests
    SERVER_VERSION = version("lean-lsp-mcp")
except PackageNotFoundError:  # pragma: no cover - local dev fallback
    SERVER_VERSION = None


logger = get_logger(__name__)


# Shared resource identifiers
TOOL_SPEC_RESOURCE_URI = f"tool-spec://lean_lsp_mcp/{TOOL_SPEC_VERSION}.json"


# Server and context
class AppContext:
    lean_project_path: str | None
    client: Any  # LeanLSPClient | None
    file_content_hashes: Dict[str, str]
    rate_limit: Dict[str, List[int]]
    project_cache: Dict[str, str]
    client_lock: Lock

    def __init__(
        self,
        *,
        lean_project_path: str | None,
        client: Any,
        file_content_hashes: Dict[str, str],
        rate_limit: Dict[str, List[int]],
        project_cache: Dict[str, str],
        client_lock: Lock,
    ) -> None:
        self.lean_project_path = lean_project_path
        self.client = client
        self.file_content_hashes = file_content_hashes
        self.rate_limit = rate_limit
        self.project_cache = project_cache
        self.client_lock = client_lock


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
    name="lean_lsp_mcp",
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

def _strip_meta_from_result(result: Any) -> Any:
    if isinstance(result, dict):
        result.pop("_meta", None)
    return result


def _disable_meta_emission(server: FastMCP) -> None:
    """Best-effort removal of FastMCP-generated `_meta` blocks."""

    # Hint to upstream helpers if such an environment flag is honoured.
    os.environ.setdefault("FASTMCP_DISABLE_META", "1")

    settings = getattr(server, "settings", None)
    if settings is not None:
        for attr_name in dir(settings):
            if "meta" not in attr_name.lower():
                continue
            value = getattr(settings, attr_name)
            try:
                if isinstance(value, bool):
                    setattr(settings, attr_name, False)
                elif isinstance(value, dict):
                    setattr(settings, attr_name, {})
                else:
                    setattr(settings, attr_name, None)
            except Exception:
                continue

    def _wrap_callable(parent: Any, name: str, func: Any) -> None:
        if getattr(func, "__lean_strip_meta__", False):
            return
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                return _strip_meta_from_result(result)

            async_wrapper.__lean_strip_meta__ = True
            try:
                setattr(parent, name, async_wrapper)
            except Exception:
                pass
        elif callable(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return _strip_meta_from_result(result)

            wrapper.__lean_strip_meta__ = True
            try:
                setattr(parent, name, wrapper)
            except Exception:
                pass

    for attr_name in dir(server):
        lower = attr_name.lower()
        if "meta" in lower or ("result" in lower and "formatter" not in lower):
            attr = getattr(server, attr_name)
            if callable(attr):
                _wrap_callable(server, attr_name, attr)

    for attr_name in (
        "result_transformers",
        "_result_transformers",
        "response_transformers",
        "response_filters",
    ):
        transforms = getattr(server, attr_name, None)
        if isinstance(transforms, list):
            transforms.append(_strip_meta_from_result)
        elif transforms is None:
            try:
                setattr(server, attr_name, [_strip_meta_from_result])
            except Exception:
                pass

    for attr_name in (
        "_build_result_meta",
        "build_result_meta",
        "_build_meta_block",
        "build_meta_block",
    ):
        if hasattr(server, attr_name):
            try:
                setattr(server, attr_name, lambda *args, **kwargs: None)
            except Exception:
                pass

    if hasattr(server, "server_version"):
        try:
            setattr(server, "server_version", None)
        except Exception:
            pass


mcp = FastMCP(**mcp_kwargs)
_disable_meta_emission(mcp)


def _set_response_format_hint(ctx: Context | None, response_format: Optional[str]) -> None:
    """Stash the caller's preferred response format on the request context."""

    if ctx is None:
        return

    request_context = getattr(ctx, "request_context", None)
    if request_context is not None:
        setattr(request_context, "_response_format_hint", response_format)


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


class LeanFileSession:
    """State bundle for working with a Lean file while the client lock is held."""

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
                response_format=response_format,
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


def success_result(
    *,
    summary: str,
    structured: Dict[str, Any] | None,
    start_time: float,
    ctx: Context | None = None,
    content: List[Dict[str, Any]] | None = None,
    response_format: str | None = None,
) -> Dict[str, Any]:
    # Avoid double-reporting the summary when callers include it in the
    # auxiliary content list.
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

    if selected_format == JSON_RESPONSE_FORMAT:
        if structured_payload is not None:
            meta = structured_payload.setdefault("_meta", {})
            if summary:
                meta.setdefault("summary", summary)
            structured_for_json = structured_payload
        else:
            structured_for_json = {"summary": summary} if summary else {}

        payload: List[Dict[str, Any]] = []
        payload.append(_json_item(structured_for_json))
        payload.extend(remaining_items)

        return mcp_result(
            content=payload or [_text_item(summary_markdown)],
            structured=structured_for_json if structured_for_json else None,
        )

    payload: List[Dict[str, Any]] = []
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

    lifespan = None
    if ctx is not None:
        lifespan = getattr(ctx.request_context, "lifespan_context", None)
    project_root = getattr(lifespan, "lean_project_path", None) if lifespan else None

    if code == ERROR_CLIENT_NOT_READY:
        _append("Run `lean_build` to initialize the Lean project and restart the Lean LSP client.")

    if code in {ERROR_CLIENT_NOT_READY, ERROR_INVALID_PATH, ERROR_BAD_REQUEST} and not project_root:
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

    derived_hints = _derive_error_hints(
        code=code,
        category=category,
        details=details,
        ctx=ctx,
    )
    combined_hints: List[str] = []
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

    if response_format is None and ctx is not None:
        request_context = getattr(ctx, "request_context", None)
        if request_context is not None:
            response_format = getattr(request_context, "_response_format_hint", None)

    selected_format = normalize_response_format(response_format)
    summary_markdown = build_markdown_summary(f"Error: {message}", detail_bullets)
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

    if selected_format == JSON_RESPONSE_FORMAT:
        structured_for_json: Dict[str, Any] = (
            structured_payload if structured_payload is not None else {}
        )
        meta = structured_for_json.setdefault("_meta", {})
        meta.setdefault("summary", f"Error: {message}")
        payload: List[Dict[str, Any]] = []
        payload.append(_json_item(structured_for_json))
        payload.extend(other_items)
        return mcp_result(
            content=payload,
            structured=structured_for_json if structured_for_json else None,
            is_error=True,
        )

    payload: List[Dict[str, Any]] = []
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
@mcp.tool(
    "lean_build",
    description="Run `lake build` (optionally `lake clean`) to refresh the Lean project and restart the cached Lean LSP client.",
    annotations=TOOL_ANNOTATIONS["lean_build"]
)
def lsp_build(ctx: Context, params: LeanBuildInput) -> Any:
    """Compile the Lean project and refresh the cached Lean language server.

    This tool wraps ``lake build`` (and optionally ``lake clean``) so automated agents
    can bring the Lean workspace into a consistent state before invoking goal-oriented
    tools. The MCP server restarts its cached Lean LSP client after building, ensuring
    diagnostics and goals reflect the freshly built artifacts.

    Parameters
    ----------
    params : LeanBuildInput
        Validated build options.
        - ``lean_project_path`` (Optional[str]): Absolute project root. Falls back to
          the path inferred from previous tool calls or the ``LEAN_PROJECT_PATH`` env.
        - ``clean`` (bool, default=False): Run ``lake clean`` first to purge build
          artifacts. Only enable when dependencies changed drastically; it slows the
          workflow and deletes incremental outputs.
        - ``response_format`` (Optional[str]): Legacy formatting hint accepted for
          compatibility. The return payload is structured independently of this flag.

    Returns
    -------
    LeanBuildResult
        ``structuredContent`` includes ``status``, ``project.path``, ``clean``, and
        ``lsp_restarted``. The textual summary states whether the build succeeded and
        when feasible attaches the captured ``lake`` log as a resource. Errors set
        ``isError=True`` with diagnostic metadata in ``details``.

    Use when
    --------
    - New files or dependencies require recompilation before inspecting goals.
    - Goal/diagnostic tools report stale data after large edits.

    Avoid when
    ----------
    - You only need read-only inspection (use the *lean_file_contents* family instead).
    - The project already compiled successfully and no new dependencies were added.

    Error handling
    --------------
    - Missing project roots raise ``ERROR_BAD_REQUEST`` with remediation guidance.
    - ``lake`` failures surface as ``ERROR_IO_FAILURE`` and include captured stdout /
      stderr snippets to speed up debugging.
    """
    started = time.perf_counter()
    lean_project_path = params.lean_project_path or None
    clean = params.clean
    response_format = params.response_format
    _set_response_format_hint(ctx, response_format)

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
            response_format=response_format,
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
                response_format=response_format,
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
                response_format=response_format,
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
            response_format=response_format,
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
            response_format=response_format,
        )


# File level tools
@mcp.tool(
    "lean_file_contents",
    description="Read Lean source text with optional 1-indexed annotations; `structured.lines` preserves numbering for downstream agents.",
    annotations=TOOL_ANNOTATIONS["lean_file_contents"]
)
def file_contents(ctx: Context, params: LeanFileContentsInput) -> Any:
    """Read a Lean source file with optional line annotations and slicing.

    Parameters
    ----------
    params : LeanFileContentsInput
        Validated file selection options.
        - ``file_path`` (str): Absolute or project-relative file path. Relative paths
          are resolved against the active Lean project root or current working dir.
        - ``annotate_lines`` (bool = True): Prepend each returned line with ``N│`` to
          help downstream tools reference positions precisely.
        - ``start_line`` (Optional[int]): 1-based line to start reading from.
        - ``line_count`` (Optional[int]): Number of lines to stream from ``start_line``.
        - ``response_format`` (Optional[str]): Legacy formatting flag. The tool always
          returns structured content plus a Markdown/text summary.

    Returns
    -------
    FileContents
        ``structuredContent`` includes file identity metadata and the requested text
        segment. When ``annotate_lines`` is true the payload exposes both raw text and
        formatted lines. Errors emit ``ERROR_INVALID_PATH`` for missing files or
        ``ERROR_BAD_REQUEST`` when pagination parameters are inconsistent.

    Use when
    --------
    - Inspecting Lean code before deciding which proof tool to call.
    - Sharing file excerpts and line numbers in evaluation workflows.

    Avoid when
    ----------
    - You only need file metadata (use other tooling).
    - You plan to modify files (this tool is strictly read-only).

    Error handling
    --------------
    - Missing or unreadable paths return actionable guidance with sanitized paths.
    - Pagination parameters outside valid ranges surface as ``ERROR_BAD_REQUEST``.
    """
    started = time.perf_counter()
    file_path = params.file_path
    annotate_lines = params.annotate_lines
    start_line = params.start_line
    line_count = params.line_count
    response_format = params.response_format  # Accept `_format` hints without altering behaviour.
    _set_response_format_hint(ctx, response_format)

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
            response_format=response_format,
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
            response_format=response_format,
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
            response_format=response_format,
        )
    except IsADirectoryError:
        message = f"Path `{sanitized_path}` is a directory. Provide a Lean source file."
        return error_result(
            message=message,
            code=ERROR_INVALID_PATH,
            details={"path": sanitized_path, "kind": "directory"},
            start_time=started,
            ctx=ctx,
            response_format=response_format,
        )

    if start_line is not None and start_line < 1:
        return error_result(
            message="`start_line` must be >= 1",
            code=ERROR_BAD_REQUEST,
            details={"start_line": start_line},
            start_time=started,
            ctx=ctx,
            response_format=response_format,
        )
    if line_count is not None and line_count < 1:
        return error_result(
            message="`line_count` must be >= 1",
            code=ERROR_BAD_REQUEST,
            details={"line_count": line_count},
            start_time=started,
            ctx=ctx,
            response_format=response_format,
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
            response_format=response_format,
        )
    if total_lines == 0 and start_line and start_line > 1:
        return error_result(
            message="`start_line` is beyond the end of the file",
            code=ERROR_BAD_REQUEST,
            details={"start_line": start_line, "total_lines": total_lines},
            start_time=started,
            ctx=ctx,
            response_format=response_format,
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
        response_format=response_format,
    )


@mcp.tool(
    "lean_diagnostic_messages",
    description="List Lean diagnostics for a file window; `structured.summary.count` reports totals and `structured.diags` encode zero-based ranges for tooling.",
    annotations=TOOL_ANNOTATIONS["lean_diagnostic_messages"]
)
def diagnostic_messages(ctx: Context, params: LeanDiagnosticMessagesInput) -> Any:
    """Collect Lean compiler diagnostics for the requested file window.

    Parameters
    ----------
    params : LeanDiagnosticMessagesInput
        Validated diagnostic query.
        - ``file_path`` (str): Absolute or project-relative path to the Lean file.
        - ``start_line`` (Optional[int]): 1-based line where diagnostics should begin.
        - ``line_count`` (Optional[int]): Number of lines to inspect from ``start_line``.
        - ``response_format`` (Optional[str]): Optional hint accepted for compatibility.

    Returns
    -------
    Diagnostics
        ``structuredContent`` includes a list of Lean diagnostic entries with
        severity, message, positions, and affected range. The textual summary
        communicates how many diagnostics matched and whether pagination applied.
        Errors set ``isError=True`` with detailed context under ``details``.

    Use when
    --------
    - You need to triage Lean errors or warnings before attempting repairs.
    - Narrowing diagnostic scope to a snippet during evaluation scenarios.

    Avoid when
    ----------
    - You require goal states or term information (use ``lean_goal`` / ``lean_term_goal``).
    - You expect the tool to write fixes; it is strictly read-only.

    Error handling
    --------------
    - Invalid line ranges return ``ERROR_BAD_REQUEST`` with offending parameters.
    - Missing files raise ``ERROR_INVALID_PATH`` and provide sanitized path hints.
    """
    started = time.perf_counter()
    file_path = params.file_path
    start_line = params.start_line
    line_count = params.line_count
    response_format = params.response_format
    _set_response_format_hint(ctx, response_format)

    try:
        with open_file_session(
            ctx,
            file_path,
            started=started,
            response_format=response_format,
        ) as file_session:
            if start_line is not None and start_line < 1:
                return error_result(
                    message="`start_line` must be >= 1",
                    code=ERROR_BAD_REQUEST,
                    details={"start_line": start_line},
                    start_time=started,
                    ctx=ctx,
                    response_format=response_format,
                )
            if line_count is not None and line_count < 1:
                return error_result(
                    message="`line_count` must be >= 1",
                    code=ERROR_BAD_REQUEST,
                    details={"line_count": line_count},
                    start_time=started,
                    ctx=ctx,
                    response_format=response_format,
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
                    response_format=response_format,
                )
            if total_lines == 0 and start_line and start_line > 1:
                return error_result(
                    message="`start_line` is beyond the end of the file",
                    code=ERROR_BAD_REQUEST,
                    details={"start_line": start_line, "total_lines": total_lines},
                    start_time=started,
                    ctx=ctx,
                    response_format=response_format,
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
                response_format=response_format,
            )
    except ToolError as exc:
        return exc.payload


@mcp.tool(
    "lean_goal",
    description="Inspect Lean proof goals at a 1-indexed file position; omitting the column probes line boundaries and returns compact `goals` entries with first-line previews.",
    annotations=TOOL_ANNOTATIONS["lean_goal"]
)
def goal(ctx: Context, params: LeanGoalInput) -> Any:
    """Retrieve the Lean proof goals at a specific file location.

    Parameters
    ----------
    params : LeanGoalInput
        Validated cursor location.
        - ``file_path`` (str): Absolute or project-relative Lean source path.
        - ``line`` (int): 1-based line number to inspect.
        - ``column`` (Optional[int]): 1-based column. Omit to let the server probe
          the most relevant position on the line.
        - ``response_format`` (Optional[str]): Formatting hint accepted for parity;
          output format is otherwise fixed.

    Returns
    -------
    Goal
        ``structuredContent`` contains the raw Lean goal state plus a rendered
        Markdown-friendly version. The summary reports whether goals exist or if the
        region is solved. Errors mark ``isError=True`` and include context (e.g.
        ``ERROR_NOT_GOAL_POSITION`` when the location contains no goal).

    Use when
    --------
    - Inspecting outstanding proof obligations at a ``sorry`` or tactic block.
    - Tracking goal evolution while iterating on snippets with ``lean_multi_attempt``.

    Avoid when
    ----------
    - The file fails to parse (run ``lean_diagnostic_messages`` first).
    - You need the expected type of a term; prefer ``lean_term_goal`` instead.

    Error handling
    --------------
    - Missing files or invalid positions surface ``ERROR_INVALID_PATH`` or
      ``ERROR_BAD_REQUEST`` with actionable details.
    - When Lean reports no goals, the tool returns a structured payload indicating
      completion rather than raising an exception.
    """
    started = time.perf_counter()
    file_path = params.file_path
    line = params.line
    column = params.column
    response_format = params.response_format
    _set_response_format_hint(ctx, response_format)

    try:
        with open_file_session(
            ctx,
            file_path,
            started=started,
            response_format=response_format,
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
                        response_format=response_format,
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
                        response_format=response_format,
                    )

                results = []
                if goal_start is not None:
                    results.append(
                        {
                            "kind": "line_start",
                            "position": {
                                "line": line - 1,
                                "character": column_start,
                            },
                            "goal": goal_to_payload(goal_start),
                        }
                    )
                if goal_end is not None:
                    results.append(
                        {
                            "kind": "line_end",
                            "position": {
                                "line": line - 1,
                                "character": column_end,
                            },
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
                    line_index: int
                    column_index: int
                    try:
                        line_value = int(raw_line) if raw_line is not None else None
                    except (TypeError, ValueError):
                        line_value = None
                    try:
                        column_value = int(raw_column) if raw_column is not None else None
                    except (TypeError, ValueError):
                        column_value = None

                    if line_value is None:
                        line_index = line
                    elif line_value >= 0:
                        line_index = line_value + 1
                    else:
                        line_index = line

                    if column_value is None:
                        column_index = column_start + 1 if kind == "line_start" else column_end + 1
                    elif column_value >= 0:
                        column_index = column_value + 1
                    else:
                        column_index = column_start + 1 if kind == "line_start" else column_end + 1

                    line_index = max(line_index, 1)
                    column_index = max(column_index, 1)
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
                    response_format=response_format,
                )

            if column < 1:
                return error_result(
                    message="Column must be >= 1",
                    code=ERROR_BAD_REQUEST,
                    details={"file": identity["relative_path"], "line": line, "column": column},
                    start_time=started,
                    ctx=ctx,
                    response_format=response_format,
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
                    response_format=response_format,
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
                response_format=response_format,
            )
    except ToolError as exc:
        return exc.payload


@mcp.tool(
    "lean_term_goal",
    description="Inspect the expected Lean term type at a 1-indexed location and expose the cleaned `rendered` snippet alongside normalized coordinates.",
    annotations=TOOL_ANNOTATIONS["lean_term_goal"]
)
def term_goal(ctx: Context, params: LeanTermGoalInput) -> Any:
    """Return the expected term type at a Lean source location.

    Parameters
    ----------
    params : LeanTermGoalInput
        Validated cursor position identical to ``lean_goal`` semantics.
        - ``file_path`` (str): Absolute or project-relative file path.
        - ``line`` (int): 1-based line number that contains the expression.
        - ``column`` (Optional[int]): 1-based column within the line. Defaults to end
          of the line when omitted.
        - ``response_format`` (Optional[str]): Formatting hint accepted for parity.

    Returns
    -------
    Goal
        ``structuredContent`` includes the rendered term type and the underlying Lean
        payload so agents can distinguish the exact position inspected. Summary text
        states whether the expected type was found. Errors set ``isError=True`` with
        specific error codes for invalid positions or missing files.

    Use when
    --------
    - Determining the expected type of a partially written term or placeholder.
    - Pairing with ``lean_goal`` to understand both hypotheses and target types.

    Avoid when
    ----------
    - You require the full goal state (hypotheses + target) – use ``lean_goal``.
    - The file has outstanding diagnostics that prevent Lean from elaborating.

    Error handling
    --------------
    - Out-of-range lines or columns emit ``ERROR_NOT_GOAL_POSITION`` or
      ``ERROR_BAD_REQUEST`` with debugging context.
    - Missing files return ``ERROR_INVALID_PATH`` with sanitized path hints.
    """
    started = time.perf_counter()
    file_path = params.file_path
    line = params.line
    column = params.column
    response_format = params.response_format
    _set_response_format_hint(ctx, response_format)

    try:
        with open_file_session(
            ctx,
            file_path,
            started=started,
            response_format=response_format,
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


@mcp.tool(
    "lean_hover_info",
    description="Retrieve Lean hover documentation for the symbol under the cursor together with `structured.infoSnippet` and nearby diagnostics.",
    annotations=TOOL_ANNOTATIONS["lean_hover_info"]
)
def hover(ctx: Context, params: LeanHoverInput) -> Any:
    """Return Lean hover information for the term under the cursor.

    Parameters
    ----------
    params : LeanHoverInput
        Validated hover position.
        - ``file_path`` (str): Absolute or project-relative Lean file path.
        - ``line`` (int): 1-based line index.
        - ``column`` (int): 1-based column pointing inside the identifier or term.
        - ``response_format`` (Optional[str]): Formatting hint; output schema is fixed.

    Returns
    -------
    Hover
        ``structuredContent`` includes symbol documentation, type information, and any
        source ranges returned by Lean. The textual summary echoes the symbol name when
        available. Errors mark ``isError=True`` with codes such as ``ERROR_NO_GOAL`` when
        nothing is hoverable at the requested position.

    Use when
    --------
    - Looking up definitions, type signatures, or docstrings for identifiers.
    - Cross-referencing Lean APIs during evaluation without opening external docs.

    Avoid when
    ----------
    - You need declaration source code (use ``lean_declaration_file``).
    - The file has syntax errors preventing Lean from producing hover info.

    Error handling
    --------------
    - Out-of-range positions emit ``ERROR_BAD_REQUEST``.
    - Missing files surface ``ERROR_INVALID_PATH`` with sanitized details.
    """
    started = time.perf_counter()
    file_path = params.file_path
    line = params.line
    column = params.column
    response_format = params.response_format
    _set_response_format_hint(ctx, response_format)

    try:
        with open_file_session(
            ctx,
            file_path,
            started=started,
            response_format=response_format,
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


@mcp.tool(
    "lean_completions",
    description="Request Lean completion suggestions at a file position.",
    annotations=TOOL_ANNOTATIONS["lean_completions"]
)
def completions(ctx: Context, params: LeanCompletionsInput) -> Any:
    """Return Lean completion suggestions for the token under the cursor.

    Parameters
    ----------
    params : LeanCompletionsInput
        Validated completion context.
        - ``file_path`` (str): Absolute or project-relative Lean file path.
        - ``line`` (int): 1-based line containing the partial identifier.
        - ``column`` (int): 1-based column within the partial token.
        - ``max_completions`` (int = 32): Maximum suggestions to return. Values above
          32 may be truncated for stability.
        - ``response_format`` (Optional[str]): Formatting hint, ignored by the tool.

    Returns
    -------
    Completions
        ``structuredContent`` lists Lean completion items with kind, text, replacement
        ranges, and documentation. The summary names the symbol being completed when
        available. Errors set ``isError=True`` with explicit codes for invalid ranges.

    Use when
    --------
    - Exploring which identifiers or namespace members are available at a location.
    - Auto-completing imports, attribute names, and tactic suggestions during proofs.

    Avoid when
    ----------
    - You need semantic info about an already-complete term (use ``lean_hover_info``).
    - The file fails to parse; run ``lean_diagnostic_messages`` first.

    Error handling
    --------------
    - Out-of-range positions emit ``ERROR_BAD_REQUEST``.
    - Missing files raise ``ERROR_INVALID_PATH`` with sanitized details.
    """
    started = time.perf_counter()
    file_path = params.file_path
    line = params.line
    column = params.column
    max_completions = params.max_completions
    response_format = params.response_format
    _set_response_format_hint(ctx, response_format)

    try:
        with open_file_session(
            ctx,
            file_path,
            started=started,
            response_format=response_format,
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


@mcp.tool(
    "lean_declaration_file",
    description="Open the Lean source file that defines a given declaration and surface declaration path metadata for navigation.",
    annotations=TOOL_ANNOTATIONS["lean_declaration_file"]
)
def declaration_file(ctx: Context, params: LeanDeclarationFileInput) -> Any:
    """Open the source file that defines a Lean symbol and return its contents.

    Parameters
    ----------
    params : LeanDeclarationFileInput
        Validated declaration lookup.
        - ``file_path`` (str): Lean source file containing a reference to ``symbol``.
        - ``symbol`` (str): Fully qualified or local name of the declaration to locate.
        - ``response_format`` (Optional[str]): Formatting hint, ignored by the tool.

    Returns
    -------
    Declaration
        ``structuredContent`` surfaces the origin location and the declaration file
        metadata. When the declaration file is small, its contents are attached as an
        inline resource to simplify follow-up analysis. Errors set ``isError=True`` and
        include ``ERROR_INVALID_PATH`` or ``ERROR_UNKNOWN`` when the symbol cannot be
        resolved by the Lean LSP.

    Use when
    --------
    - You need to read the implementation of a referenced theorem or definition.
    - Preparing context for prompts that require the declaration source text.

    Avoid when
    ----------
    - You only need quick documentation (use ``lean_hover_info``).
    - Symbols resolve to generated or external library code that is not accessible.

    Error handling
    --------------
    - Missing symbols return actionable error details with sanitized file paths.
    - Non-existent files produce ``ERROR_INVALID_PATH`` listing the offending target.
    """

    started = time.perf_counter()
    file_path = params.file_path
    symbol = params.symbol
    response_format = params.response_format
    _set_response_format_hint(ctx, response_format)

    try:
        with open_file_session(
            ctx,
            file_path,
            started=started,
            response_format=response_format,
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

            decl_line = max(position["line"] - 1, 0)
            decl_col = max(position["column"] - 1, 0)
            declaration = file_session.client.get_declarations(
                file_session.rel_path, decl_line, decl_col
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


@mcp.tool(
    "lean_multi_attempt",
    description="Evaluate multiple Lean snippets at a line and compare diagnostics/goals.",
    annotations=TOOL_ANNOTATIONS["lean_multi_attempt"]
)
def multi_attempt(ctx: Context, params: LeanMultiAttemptInput) -> Any:
    """Evaluate multiple Lean snippets at one location and compare outcomes.

    Parameters
    ----------
    params : LeanMultiAttemptInput
        Validated snippet specification.
        - ``file_path`` (str): Absolute or project-relative Lean file path.
        - ``line`` (int): 1-based line where each snippet is applied.
        - ``snippets`` (List[str]): One or more tactic/code variants to test. Each
          snippet is inserted as-is on the target line.
        - ``response_format`` (Optional[str]): Formatting hint accepted for parity.

    Returns
    -------
    MultiAttempt
        ``structuredContent`` contains per-snippet diagnostics and goal payloads so
        agents can contrast options. The textual summary lists how many diagnostics
        fired for each snippet. Errors set ``isError=True`` when inputs are invalid.

    Use when
    --------
    - Comparing alternative tactic strategies before editing files.
    - Investigating how slight snippet changes affect goals and diagnostics.

    Avoid when
    ----------
    - You only have a single snippet to test (use ``lean_run_code`` or direct edits).
    - Snippets span multiple lines; the tool supports single-line inserts only.

    Error handling
    --------------
    - Missing or empty snippet arrays return ``ERROR_BAD_REQUEST``.
    - File resolution errors surface ``ERROR_INVALID_PATH`` with sanitized metadata.
    """

    started = time.perf_counter()
    file_path = params.file_path
    line = params.line
    snippets = list(params.snippets)
    response_format = params.response_format
    _set_response_format_hint(ctx, response_format)

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
            response_format=response_format,
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


@mcp.tool(
    "lean_run_code",
    description="Execute an isolated Lean snippet with standalone diagnostics.",
    annotations=TOOL_ANNOTATIONS["lean_run_code"]
)
def run_code(ctx: Context, params: LeanRunCodeInput) -> Any:
    """Execute a self-contained Lean snippet in an isolated buffer.

    Parameters
    ----------
    params : LeanRunCodeInput
        Validated snippet payload.
        - ``code`` (str): Complete Lean program, including necessary imports.
        - ``response_format`` (Optional[str]): Formatting hint accepted for parity.

    Returns
    -------
    RunCode
        ``structuredContent`` contains diagnostics emitted by Lean along with rendered
        messages and severity summaries. The textual summary states whether the snippet
        compiled successfully. Errors set ``isError=True`` with contextual metadata.

    Use when
    --------
    - Testing small standalone snippets that should not touch on-disk files.
    - Exploring language features without modifying the active project sources.

    Avoid when
    ----------
    - You need to inspect results tied to existing files (use file-scoped tools).
    - The snippet relies on project state (e.g. local definitions); prefer editing
      files and running ``lean_diagnostic_messages`` instead.

    Error handling
    --------------
    - Missing project roots raise ``ERROR_BAD_REQUEST`` with remediation steps.
    - Lean compiler failures are returned as structured diagnostics, not exceptions.
    """
    started = time.perf_counter()
    code = params.code
    response_format = params.response_format
    _set_response_format_hint(ctx, response_format)

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


def _render_tool_spec_payload() -> tuple[Dict[str, Any], str]:
    spec = build_tool_spec()
    spec_json = json.dumps(spec, indent=2, ensure_ascii=False)
    return spec, spec_json


@mcp.tool(
    "lean_tool_spec",
    description="Publish structured metadata for all Lean MCP tools, including inputs, annotations, and rate limits.",
    annotations=TOOL_ANNOTATIONS["lean_tool_spec"]
)
def tool_spec(ctx: Context, params: LeanToolSpecInput) -> Any:
    """Return the published Lean MCP tool specification for evaluation harnesses.

    Parameters
    ----------
    params : LeanToolSpecInput
        Optional request metadata.
        - ``response_format`` (Optional[str]): Legacy hint accepted for compatibility.

    Returns
    -------
    ToolSpecSummary
        ``structuredContent`` contains the result of
        ``lean_lsp_mcp.tool_spec.build_tool_spec`` including version, tool definitions,
        and response summaries. The human-readable content includes a short summary and
        an attached JSON resource whose URI encodes the spec version for caching.
        For direct module access use
        ``lean_lsp_mcp.tool_spec.build_tool_spec`` inside the Python package.

    Use when
    --------
    - Quickly listing available tools from an MCP client without calling ``listTools``.
    - Feeding the spec into the mcp-builder evaluation harness as part of setup.

    Avoid when
    ----------
    - You require the full JSON schema (call the module directly to stream the file).
    - You only need high-level docs; check the README workflows instead.
    """
    response_format = params.response_format
    _set_response_format_hint(ctx, response_format)
    started = time.perf_counter()
    spec, spec_json = _render_tool_spec_payload()
    summary = "Lean MCP tool specification ready."
    content_items = [
        _text_item(summary),
        _resource_item(TOOL_SPEC_RESOURCE_URI, spec_json, mime_type="application/json"),
    ]
    return success_result(
        summary=summary,
        structured=spec,
        content=content_items,
        start_time=started,
        ctx=ctx,
    )


# Register the tool-spec resource when FastMCP exposes a resource API (test stubs may not).
def _tool_spec_resource_impl() -> str:
    """Return the tool specification JSON payload."""
    _, spec_json = _render_tool_spec_payload()
    return spec_json

_resource_reg = getattr(mcp, "resource", None)
if callable(_resource_reg):  # pragma: no cover - exercised in real server runtime
    _tool_spec_resource_impl = _resource_reg(
        TOOL_SPEC_RESOURCE_URI,
        title="Lean MCP Tool Specification",
        description=(
            "Structured metadata for all Lean MCP tools, including schemas and annotations."
        ),
        mime_type="application/json",
    )(_tool_spec_resource_impl)


@mcp.tool(
    "lean_leansearch",
    description="Query leansearch.net for Lean lemmas/theorems; responses list fully qualified names in `structured.names` (3 req/30s rate limit).",
    annotations=TOOL_ANNOTATIONS["lean_leansearch"]
)
@rate_limited("leansearch", max_requests=3, per_seconds=30)
def leansearch(ctx: Context, params: LeanSearchInput) -> Any:
    """Query leansearch.net for theorems, lemmas, and tactics relevant to a prompt.

    Parameters
    ----------
    params : LeanSearchInput
        Validated search parameters.
        - ``query`` (str): Free-text, Lean syntax, or hybrid search string.
        - ``num_results`` (int = 5): Maximum number of results to request (>=1).
        - ``response_format`` (Optional[str]): Formatting hint, ignored by this tool.

    Returns
    -------
    SearchResults
        ``structuredContent`` lists matched entries with names, statements, relevance
        scores, and URIs. The summary highlights how many matches were returned or if
        none were found. Rate limits are enforced at 3 requests per 30 seconds.

    Use when
    --------
    - You need candidate lemmas or theorems to apply in a Lean proof.
    - Bootstrapping evaluation questions that require external theorem discovery.

    Avoid when
    ----------
    - You must search local project files (use other file-centric tools).
    - The query can be answered via Lean's standard library alone (hover/goal might be enough).

    Error handling
    --------------
    - Empty responses return ``ERROR_BAD_REQUEST`` with guidance to refine the query.
    - Network or parsing issues surface as ``ERROR_UNKNOWN`` including server messages.
    """
    started = time.perf_counter()
    query = params.query
    num_results = params.num_results
    response_format = params.response_format
    _set_response_format_hint(ctx, response_format)

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


@mcp.tool(
    "lean_loogle",
    description="Query loogle.lean-lang.org for Lean declarations by name/pattern; compact responses populate `structured.names` (3 req/30s).",
    annotations=TOOL_ANNOTATIONS["lean_loogle"]
)
@rate_limited("loogle", max_requests=3, per_seconds=30)
def loogle(ctx: Context, params: LoogleSearchInput) -> Any:
    """Search loogle.lean-lang.org for Lean lemmas by name, constants, or patterns.

    Parameters
    ----------
    params : LoogleSearchInput
        Validated search arguments.
        - ``query`` (str): Raw loogle query supporting constants, patterns, and
          conclusion matching.
        - ``num_results`` (int = 8): Maximum number of hits to fetch.
        - ``response_format`` (Optional[str]): Formatting hint; output schema is fixed.

    Returns
    -------
    SearchResults
        ``structuredContent`` contains matched lemma names, statements, and proof state
        hints. The summary reports the number of hits and includes a quick preview.
        Calls are rate-limited to 3 per 30 seconds.

    Use when
    --------
    - You recall part of a lemma name or constant and need the full theorem.
    - Filtering results by expression patterns or conclusion shape.

    Avoid when
    ----------
    - The query is purely natural language (``lean_leansearch`` handles fuzzy search).
    - You need project-local lemmas not indexed by loogle.

    Error handling
    --------------
    - Empty hits return ``ERROR_UNKNOWN`` with guidance to tighten or broaden queries.
    - HTTP or parsing errors set ``ERROR_UNKNOWN`` with the upstream error text.
    """
    started = time.perf_counter()
    query = params.query
    num_results = params.num_results
    response_format = params.response_format
    _set_response_format_hint(ctx, response_format)
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


@mcp.tool(
    "lean_state_search",
    description="Fetch goal-based lemma suggestions from premise-search.com (3 req/30s).",
    annotations=TOOL_ANNOTATIONS["lean_state_search"]
)
@rate_limited("lean_state_search", max_requests=3, per_seconds=30)
def state_search(ctx: Context, params: LeanStateSearchInput) -> Any:
    """Query premise-search.com for lemmas relevant to the current goal state.

    Parameters
    ----------
    params : LeanStateSearchInput
        Validated state-search request.
        - ``file_path`` (str): Absolute or project-relative Lean file path.
        - ``line`` (int): 1-based line containing the goal.
        - ``column`` (int): 1-based column used to extract the goal (first goal only).
        - ``num_results`` (int = 5): Maximum suggestions to return.
        - ``response_format`` (Optional[str]): Formatting hint; output schema is fixed.

    Environment Variables
    ---------------------
    - ``LEAN_STATE_SEARCH_URL`` (Optional[str]): Base URL for premise-search API.
      Defaults to "https://premise-search.com". For production use, consider
      self-hosting: https://github.com/ruc-ai4math/LeanStateSearch
      Example: ``LEAN_STATE_SEARCH_URL=http://localhost:3000``
    - ``LEAN_STATE_SEARCH_REV`` (Optional[str]): Mathlib revision to query against.
      Defaults to "v4.16.0". The public instance may have limited revision support.
      Self-hosted instances can index any revision.
      Example: ``LEAN_STATE_SEARCH_REV=v4.18.0``

    Returns
    -------
    SearchResults
        ``structuredContent`` includes candidate lemmas with names and statements
        ranked by relevance. The summary reports how many suggestions were produced.
        Tool calls are rate-limited to 3 per 30 seconds.

    Use when
    --------
    - You have an active goal and want external guidance on which lemmas to try.
    - Generating evaluation tasks that require multi-hop reasoning with external hints.

    Avoid when
    ----------
    - You need keyword search over lemma names (use ``lean_loogle`` or ``lean_leansearch``).
    - The file has compilation errors preventing Lean from computing goals.

    Error handling
    --------------
    - Missing goals return ``ERROR_NO_GOAL`` with the inspected position.
    - External service failures surface as ``ERROR_UNKNOWN`` with upstream detail.
    """
    started = time.perf_counter()
    file_path = params.file_path
    line = params.line
    column = params.column
    num_results = params.num_results
    response_format = params.response_format
    _set_response_format_hint(ctx, response_format)

    try:
        with open_file_session(
            ctx,
            file_path,
            started=started,
            response_format=response_format,
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

    f_line = format_line(file_contents, line, column)
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

    # premise-search.com API changed to GET with query parameters
    # Default to v4.16.0 but allow override via environment variable
    # Note: Public instance may have limited revision support; self-hosting recommended
    mathlib_rev = os.getenv("LEAN_STATE_SEARCH_REV", "v4.16.0")
    goal_text = goal_state["goals"][0]

    # Check for custom URL (self-hosted instance)
    base_url = os.getenv("LEAN_STATE_SEARCH_URL", "https://premise-search.com")
    if not base_url.startswith("http"):
        base_url = f"https://{base_url}"
    base_url = base_url.rstrip("/")

    try:
        # Encode goal state for URL
        query_encoded = urllib.parse.quote(goal_text)
        url = (
            f"{base_url}/api/search"
            f"?query={query_encoded}"
            f"&results={num_results}"
            f"&rev={mathlib_rev}"
        )

        headers = {"User-Agent": "lean-lsp-mcp/0.1"}
        req = urllib.request.Request(url, headers=headers, method="GET")

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
            "query": {
                "state": goal_text,
                "limit": num_results,
                "lineSnippet": f_line,
                "rev": mathlib_rev,
            },
            "names": names,
        }
        summary = f"{len(results)} results"
        return success_result(
            summary=summary,
            structured=structured,
            start_time=started,
            ctx=ctx,
        )
    except urllib.error.HTTPError as e:
        # Handle specific HTTP errors with actionable guidance
        if e.code == 405:
            return error_result(
                message="premise-search.com API method not allowed",
                code=ERROR_UNKNOWN,
                category="lean_state_search",
                details={
                    "error": "The API may have changed. Please report this issue.",
                    "http_code": 405,
                    "file": identity["relative_path"],
                },
                hints=[
                    "The premise-search.com API interface may have been updated.",
                    "Check https://premise-search.com for current API documentation.",
                    "Consider using a self-hosted instance via LEAN_STATE_SEARCH_URL."
                ],
                start_time=started,
                ctx=ctx,
            )
        elif e.code == 400:
            # Try to parse error message from API
            try:
                error_data = json.loads(e.read().decode("utf-8"))
                error_msg = error_data.get("error", str(e))
            except:
                error_msg = str(e)
            
            # Check if it's a revision issue
            is_revision_error = "revision" in error_msg.lower()
            
            return error_result(
                message=f"Premise search API error: {error_msg}",
                code=ERROR_BAD_REQUEST,
                category="lean_state_search",
                details={
                    "error": error_msg,
                    "http_code": 400,
                    "file": identity["relative_path"],
                    "rev": mathlib_rev,
                    "url": base_url,
                },
                hints=[
                    f"The Mathlib revision '{mathlib_rev}' may not be available on {base_url}.",
                    "The public premise-search.com instance may have limited revision support.",
                    "Recommended: Self-host for production use: https://github.com/ruc-ai4math/LeanStateSearch",
                    "Set LEAN_STATE_SEARCH_URL to your self-hosted instance URL.",
                    "Alternative: Use lean_leansearch or lean_loogle for theorem search.",
                ] if is_revision_error else [
                    "Check that your goal state is properly formatted.",
                    "Ensure the query parameter contains valid Lean syntax.",
                    f"API endpoint: {base_url}/api/search"
                ],
                start_time=started,
                ctx=ctx,
            )
        return error_result(
            message=f"premise-search.com HTTP error {e.code}",
            code=ERROR_UNKNOWN,
            category="lean_state_search",
            details={
                "error": str(e),
                "http_code": e.code,
                "line": line,
                "column": column,
                "file": identity["relative_path"],
            },
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


@mcp.tool(
    "lean_hammer_premise",
    description="Retrieve hammer premise suggestions for the active goal (3 req/30s).",
    annotations=TOOL_ANNOTATIONS["lean_hammer_premise"]
)
@rate_limited("hammer_premise", max_requests=3, per_seconds=30)
def hammer_premise(ctx: Context, params: LeanHammerPremiseInput) -> Any:
    """Retrieve supporting premises from a Lean hammer server for the current goal.

    Parameters
    ----------
    params : LeanHammerPremiseInput
        Validated hammer query.
        - ``file_path`` (str): Absolute or project-relative Lean file path.
        - ``line`` (int): 1-based line containing the goal.
        - ``column`` (int): 1-based column to anchor the goal extraction.
        - ``num_results`` (int = 32): Maximum number of premises to request.
        - ``response_format`` (Optional[str]): Formatting hint; output schema is fixed.

    Returns
    -------
    SearchResults
        ``structuredContent`` lists premise candidates with scores and references to
        the Lean library. Summaries indicate how many premises were returned. Rate
        limits enforce 3 requests per 30 seconds.

    Use when
    --------
    - You need a curated list of premises to guide a proof search or automation step.
    - Designing evaluation tasks that require combining multiple helper results.

    Avoid when
    ----------
    - No goal is available at the cursor (run ``lean_goal`` first).
    - The hammer backend is unavailable; prefer ``lean_state_search`` as fallback.

    Error handling
    --------------
    - Missing goals return ``ERROR_NO_GOAL`` with file/line context.
    - HTTP or parsing issues surface as ``ERROR_UNKNOWN`` with upstream details.
    """
    started = time.perf_counter()
    file_path = params.file_path
    line = params.line
    column = params.column
    num_results = params.num_results
    response_format = params.response_format
    _set_response_format_hint(ctx, response_format)

    try:
        with open_file_session(
            ctx,
            file_path,
            started=started,
            response_format=response_format,
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
