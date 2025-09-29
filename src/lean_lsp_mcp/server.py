import os
import re
import time
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager, contextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
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
from leanclient import LeanLSPClient, DocumentContentChange

from lean_lsp_mcp.client_utils import setup_client_for_file, startup_client
from lean_lsp_mcp.file_utils import (
    get_file_contents,
    resolve_absolute_file_path,
    update_file,
)
from lean_lsp_mcp.instructions import INSTRUCTIONS
from lean_lsp_mcp.tool_spec import TOOL_SPEC_VERSION, build_tool_spec
from lean_lsp_mcp.schema import make_response
from lean_lsp_mcp.schema_types import (
    ERROR_BAD_REQUEST,
    ERROR_CLIENT_NOT_READY,
    ERROR_INVALID_PATH,
    ERROR_NO_GOAL,
    ERROR_NOT_GOAL_POSITION,
    ERROR_RATE_LIMIT,
)
from lean_lsp_mcp.utils import (
    OutputCapture,
    compute_pagination,
    diagnostics_to_entries,
    extract_range,
    filter_diagnostics_by_position,
    find_start_position,
    format_diagnostics,
    format_goal,
    format_line,
    goal_to_payload,
    normalize_range,
    OptionalTokenVerifier,
)


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
    dependencies=["leanclient"],
    lifespan=app_lifespan
)

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

def ok_response(data, meta=None, legacy_text=None):
    formatter = legacy_text
    return make_response("ok", data=data, meta=meta, legacy_formatter=formatter)


def error_response(
    message: str,
    data=None,
    meta=None,
    legacy_text=None,
    *,
    code: str | None = None,
):
    payload = {"message": message}
    if data:
        payload.update(data)
    formatter = legacy_text or message
    meta_payload = dict(meta) if meta else {}
    if code:
        existing_error = meta_payload.get("error")
        if isinstance(existing_error, dict):
            existing_error.setdefault("code", code)
        else:
            meta_payload["error"] = {"code": code}
    return make_response(
        "error",
        data=payload,
        meta=meta_payload or None,
        legacy_formatter=formatter,
    )


# Rate limiting: n requests per m seconds
def rate_limited(category: str, max_requests: int, per_seconds: int):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ctx: Context | None = kwargs.get("ctx")
            if ctx is None:
                if not args:
                    raise TypeError(
                        "rate_limited tools must receive `ctx` as the first argument"
                    )
                ctx = args[0]

            rate_limit = ctx.request_context.lifespan_context.rate_limit
            current_time = int(time.time())
            rate_limit[category] = [
                timestamp
                for timestamp in rate_limit[category]
                if timestamp > current_time - per_seconds
            ]
            if len(rate_limit[category]) >= max_requests:
                message = (
                    f"Tool limit exceeded: {max_requests} requests per {per_seconds} s."
                    " Try again later."
                )
                return error_response(
                    message,
                    data={"category": category},
                    meta={
                        "rate_limit": {
                            "max_requests": max_requests,
                            "per_seconds": per_seconds,
                        }
                    },
                    legacy_text=message,
                    code=ERROR_RATE_LIMIT,
                )
            rate_limit[category].append(current_time)
            return func(*args, **kwargs)

        original_doc = wrapper.__doc__ or ""
        wrapper.__doc__ = f"Limit: {max_requests}req/{per_seconds}s. " + original_doc
        return wrapper

    return decorator


# Project level tools
@mcp.tool("lean_build")
def lsp_build(ctx: Context, lean_project_path: str = None, clean: bool = False) -> Any:
    """Build the Lean project and restart the LSP Server.

    Use only if needed (e.g. new imports).

    Args:
        lean_project_path (str, optional): Path to the Lean project. If not provided, it will be inferred from previous tool calls.
        clean (bool, optional): Run `lake clean` before building. Attention: Only use if it is really necessary! It can take a long time! Defaults to False.

    Returns:
        str: Build output or error msg
    """
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
        return error_response(message)

    build_output = ""
    try:
        client: LeanLSPClient = ctx.request_context.lifespan_context.client
        if client:
            client.close()
            ctx.request_context.lifespan_context.file_content_hashes.clear()

        if clean:
            subprocess.run(["lake", "clean"], cwd=lean_project_path, check=False)
            logger.info("Ran `lake clean`")

        with OutputCapture() as output:
            client = LeanLSPClient(
                lean_project_path,
                initial_build=True,
                print_warnings=False,
            )
        logger.info("Built project and re-started LSP client")

        ctx.request_context.lifespan_context.client = client
        build_output = output.get_output()
        return ok_response(
            data={
                "project_path": lean_project_path,
                "output": build_output,
                "clean": clean,
            },
            meta={"operation": "build"},
            legacy_text=build_output,
        )
    except Exception as e:
        error_text = f"Error during build:\n{str(e)}\n{build_output}"
        return error_response(
            "Error during build",
            data={
                "error": str(e),
                "output": build_output,
                "project_path": lean_project_path,
                "clean": clean,
            },
            meta={"operation": "build"},
            legacy_text=error_text,
        )


# File level tools
@mcp.tool("lean_file_contents")
def file_contents(
    ctx: Context,
    file_path: str,
    annotate_lines: bool = True,
    start_line: Optional[int] = None,
    line_count: Optional[int] = None,
) -> Any:
    """Get the text contents of a Lean file, optionally with line numbers.

    Args:
        file_path (str): Abs path to Lean file
        annotate_lines (bool, optional): Annotate lines with line numbers. Defaults to True.
        start_line (int, optional): 1-based line to start from.
        line_count (int, optional): Number of lines to return from ``start_line``.

    Returns:
        str: File content or error msg
    """
    abs_path = resolve_absolute_file_path(ctx, file_path)
    if abs_path is None:
        message = (
            f"File `{file_path}` does not exist. Please check the path and try again."
        )
        return error_response(message, data={"path": file_path}, code=ERROR_INVALID_PATH)

    try:
        data = get_file_contents(abs_path)
    except FileNotFoundError:
        message = (
            f"File `{abs_path}` does not exist. Please check the path and try again."
        )
        return error_response(message, data={"path": abs_path}, code=ERROR_INVALID_PATH)

    if start_line is not None and start_line < 1:
        return error_response(
            "`start_line` must be >= 1",
            data={"start_line": start_line},
            code=ERROR_BAD_REQUEST,
        )
    if line_count is not None and line_count < 1:
        return error_response(
            "`line_count` must be >= 1",
            data={"line_count": line_count},
            code=ERROR_BAD_REQUEST,
        )

    lines = data.split("\n")
    total_lines = len(lines)

    if start_line and total_lines and start_line > total_lines:
        return error_response(
            "`start_line` is beyond the end of the file",
            data={"start_line": start_line, "total_lines": total_lines},
            code=ERROR_BAD_REQUEST,
        )
    if total_lines == 0 and start_line and start_line > 1:
        return error_response(
            "`start_line` is beyond the end of the file",
            data={"start_line": start_line, "total_lines": total_lines},
            code=ERROR_BAD_REQUEST,
        )

    start, end, pagination_meta = compute_pagination(
        total_lines if total_lines else 0, start_line, line_count
    )

    if total_lines == 0:
        slice_lines: List[str] = []
    else:
        slice_lines = lines[start - 1 : end]

    meta = {"pagination": pagination_meta}

    if annotate_lines:
        payload_lines = [
            {"number": start + idx, "text": line}
            for idx, line in enumerate(slice_lines)
        ]
        payload = {
            "path": abs_path,
            "annotated": True,
            "lines": payload_lines,
        }
        max_digits = len(str(total_lines if total_lines else 1))
        annotated = "".join(
            f"{entry['number']}{' ' * (max_digits - len(str(entry['number'])))}: {entry['text']}\n"
            for entry in payload_lines
        )
        return ok_response(payload, meta=meta, legacy_text=annotated)

    slice_text = "\n".join(slice_lines)
    payload = {
        "path": abs_path,
        "annotated": False,
        "contents": slice_text,
        "start_line": start,
    }
    return ok_response(payload, meta=meta, legacy_text=slice_text)


@mcp.tool("lean_diagnostic_messages")
def diagnostic_messages(
    ctx: Context,
    file_path: str,
    start_line: Optional[int] = None,
    line_count: Optional[int] = None,
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
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        message = "Invalid Lean file path: Unable to start LSP server or load file"
        return error_response(
            message,
            data={"file_path": file_path},
            code=ERROR_INVALID_PATH,
        )

    if start_line is not None and start_line < 1:
        return error_response(
            "`start_line` must be >= 1",
            data={"start_line": start_line},
            code=ERROR_BAD_REQUEST,
        )
    if line_count is not None and line_count < 1:
        return error_response(
            "`line_count` must be >= 1",
            data={"line_count": line_count},
            code=ERROR_BAD_REQUEST,
        )

    with client_session(ctx) as client:
        if client is None:
            return error_response(
                "Lean client is not available. Run another tool to initialize the project first.",
                code=ERROR_CLIENT_NOT_READY,
            )

        content = update_file(ctx, rel_path)
        total_lines = len(content.splitlines())
        if total_lines and start_line and start_line > total_lines:
            return error_response(
                "`start_line` is beyond the end of the file",
                data={"start_line": start_line, "total_lines": total_lines},
                code=ERROR_BAD_REQUEST,
            )
        if total_lines == 0 and start_line and start_line > 1:
            return error_response(
                "`start_line` is beyond the end of the file",
                data={"start_line": start_line, "total_lines": total_lines},
                code=ERROR_BAD_REQUEST,
            )

        start, end, pagination_meta = compute_pagination(
            total_lines, start_line, line_count
        )
        start_idx = max(0, start - 1)
        end_idx = max(0, end - 1)

        diagnostics = client.get_diagnostics(rel_path)

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

        legacy = format_diagnostics(filtered)
        payload = {
            "file": rel_path,
            "diagnostics": diagnostics_to_entries(filtered),
        }
        return ok_response(
            payload,
            meta={"pagination": pagination_meta},
            legacy_text=legacy,
        )


@mcp.tool("lean_goal")
def goal(ctx: Context, file_path: str, line: int, column: Optional[int] = None) -> Any:
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
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        message = "Invalid Lean file path: Unable to start LSP server or load file"
        return error_response(
            message,
            data={"file_path": file_path, "line": line, "column": column},
            code=ERROR_INVALID_PATH,
        )

    with client_session(ctx) as client:
        if client is None:
            return error_response(
                "Lean client is not available. Run another tool to initialize the project first.",
                data={"file": rel_path, "line": line, "column": column},
                code=ERROR_CLIENT_NOT_READY,
            )

        content = update_file(ctx, rel_path)

        if column is None:
            lines = content.splitlines()
            if line < 1 or line > len(lines):
                message = "Line number out of range. Try elsewhere?"
                return error_response(
                    message,
                    data={"file": rel_path, "line": line},
                    legacy_text=message,
                    code=ERROR_NOT_GOAL_POSITION,
                )
            column_end = len(lines[line - 1])
            column_start = next(
                (i for i, c in enumerate(lines[line - 1]) if not c.isspace()), 0
            )
            goal_start = client.get_goal(rel_path, line - 1, column_start)
            goal_end = client.get_goal(rel_path, line - 1, column_end)

            if goal_start is None and goal_end is None:
                line_text = lines[line - 1]
                message = f"No goals on line:\n{line_text}\nTry another line?"
                return error_response(
                    "No goals on line",
                    data={"file": rel_path, "line": line, "line_text": line_text},
                    legacy_text=message,
                    code=ERROR_NO_GOAL,
                )

            payload = {
                "file": rel_path,
                "request": {"line": line, "column": None},
                "line": lines[line - 1],
                "results": [
                    {"position": "line_start", "goal": goal_to_payload(goal_start)},
                    {"position": "line_end", "goal": goal_to_payload(goal_end)},
                ],
            }
            start_text = format_goal(goal_start, "No goals at line start.")
            end_text = format_goal(goal_end, "No goals at line end.")
            legacy = (
                "Goals on line:\n"
                f"{lines[line - 1]}\n"
                f"Before:\n{start_text}\nAfter:\n{end_text}"
            )
            return ok_response(payload, legacy_text=legacy)

        goal = client.get_goal(rel_path, line - 1, column - 1)
        f_goal = format_goal(goal, "Not a valid goal position. Try elsewhere?")
        f_line = format_line(content, line, column)
        if goal is None:
            message = f"Goals at:\n{f_line}\n{f_goal}"
            return error_response(
                "Not a valid goal position",
                data={"file": rel_path, "line": line, "column": column},
                legacy_text=message,
                code=ERROR_NOT_GOAL_POSITION,
            )

        payload = {
            "file": rel_path,
            "request": {"line": line, "column": column},
            "position": {"line": line, "column": column},
            "goal": goal_to_payload(goal),
            "line": format_line(content, line, None),
            "line_with_cursor": f_line,
        }
        legacy = f"Goals at:\n{f_line}\n{f_goal}"
        return ok_response(payload, legacy_text=legacy)


@mcp.tool("lean_term_goal")
def term_goal(
    ctx: Context, file_path: str, line: int, column: Optional[int] = None
) -> Any:
    """Get the expected type (term goal) at a specific location in a Lean file.

    Args:
        file_path (str): Abs path to Lean file
        line (int): Line number (1-indexed)
        column (int, optional): Column number (1-indexed). Defaults to None => end of line.

    Returns:
        str: Expected type or error msg
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        message = "Invalid Lean file path: Unable to start LSP server or load file"
        return error_response(
            message,
            data={"file_path": file_path, "line": line, "column": column},
            code=ERROR_INVALID_PATH,
        )

    with client_session(ctx) as client:
        if client is None:
            return error_response(
                "Lean client is not available. Run another tool to initialize the project first.",
                data={"file": rel_path, "line": line, "column": column},
                code=ERROR_CLIENT_NOT_READY,
            )

        content = update_file(ctx, rel_path)
        if column is None:
            lines = content.splitlines()
            if line < 1 or line > len(lines):
                message = "Line number out of range. Try elsewhere?"
                return error_response(
                    message,
                    data={"file": rel_path, "line": line},
                    legacy_text=message,
                    code=ERROR_NOT_GOAL_POSITION,
                )
            column = len(lines[line - 1])

        term_goal = client.get_term_goal(rel_path, line - 1, column - 1)
        f_line = format_line(content, line, column)
        if term_goal is None:
            message = f"Not a valid term goal position:\n{f_line}\nTry elsewhere?"
            return error_response(
                "Not a valid term goal position",
                data={"file": rel_path, "line": line, "column": column},
                legacy_text=message,
                code=ERROR_NOT_GOAL_POSITION,
            )
        rendered = term_goal.get("goal", None)
        if rendered is not None:
            rendered = rendered.replace("```lean\n", "").replace("\n```", "")
        payload = {
            "file": rel_path,
            "position": {"line": line, "column": column},
            "line": format_line(content, line, None),
            "line_with_cursor": f_line,
            "rendered": rendered,
            "raw": term_goal,
        }
        legacy = f"Term goal at:\n{f_line}\n{rendered or 'No term goal found.'}"
        return ok_response(payload, legacy_text=legacy)


@mcp.tool("lean_hover_info")
def hover(ctx: Context, file_path: str, line: int, column: int) -> Any:
    """Get hover info (docs for syntax, variables, functions, etc.) at a specific location in a Lean file.

    Args:
        file_path (str): Abs path to Lean file
        line (int): Line number (1-indexed)
        column (int): Column number (1-indexed). Make sure to use the start or within the term, not the end.

    Returns:
        str: Hover info or error msg
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        message = "Invalid Lean file path: Unable to start LSP server or load file"
        return error_response(
            message,
            data={"file_path": file_path, "line": line, "column": column},
            code=ERROR_INVALID_PATH,
        )

    with client_session(ctx) as client:
        if client is None:
            return error_response(
                "Lean client is not available. Run another tool to initialize the project first.",
                data={"file": rel_path, "line": line, "column": column},
                code=ERROR_CLIENT_NOT_READY,
            )

        file_content = update_file(ctx, rel_path)
        hover_info = client.get_hover(rel_path, line - 1, column - 1)
        if hover_info is None:
            f_line = format_line(file_content, line, column)
            message = f"No hover information at position:\n{f_line}\nTry elsewhere?"
            return error_response(
                "No hover information",
                data={"file": rel_path, "line": line, "column": column},
                legacy_text=message,
            )

        # Get the symbol and the hover information
        h_range = hover_info.get("range")
        symbol = extract_range(file_content, h_range)
        info = hover_info["contents"].get("value", "No hover information available.")
        info = info.replace("```lean\n", "").replace("\n```", "").strip()

        # Add diagnostics if available
        diagnostics = client.get_diagnostics(rel_path)
        filtered = filter_diagnostics_by_position(diagnostics, line - 1, column - 1)

        payload = {
            "file": rel_path,
            "position": {"line": line, "column": column},
            "symbol": symbol,
            "range": normalize_range(h_range),
            "info": info,
            "diagnostics": diagnostics_to_entries(filtered),
        }
        legacy = f"Hover info `{symbol}`:\n{info}"
        if filtered:
            legacy += "\n\nDiagnostics\n" + "\n".join(format_diagnostics(filtered))
        return ok_response(payload, legacy_text=legacy)


@mcp.tool("lean_completions")
def completions(
    ctx: Context, file_path: str, line: int, column: int, max_completions: int = 32
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
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        message = "Invalid Lean file path: Unable to start LSP server or load file"
        return error_response(
            message,
            data={"file_path": file_path, "line": line, "column": column},
            code=ERROR_INVALID_PATH,
        )
    with client_session(ctx) as client:
        if client is None:
            return error_response(
                "Lean client is not available. Run another tool to initialize the project first.",
                data={"file": rel_path, "line": line, "column": column},
                code=ERROR_CLIENT_NOT_READY,
            )

        content = update_file(ctx, rel_path)

        completions = client.get_completions(rel_path, line - 1, column - 1)
        formatted = [c["label"] for c in completions if "label" in c]
        f_line = format_line(content, line, column)

        if not formatted:
            message = f"No completions at position:\n{f_line}\nTry elsewhere?"
            return error_response(
                "No completions",
                data={"file": rel_path, "line": line, "column": column},
                legacy_text=message,
            )

        # Find the sort term: The last word/identifier before the cursor
        lines = content.splitlines()
        prefix = ""
        if 0 < line <= len(lines):
            text_before_cursor = lines[line - 1][: column - 1] if column > 0 else ""
            if not text_before_cursor.endswith("."):
                prefix = re.split(r"[\s()\[\]{},:;.]+", text_before_cursor)[-1].lower()

        # Sort completions: prefix matches first, then contains, then alphabetical
        if prefix:

            def sort_key(item):
                item_lower = item.lower()
                if item_lower.startswith(prefix):
                    return (0, item_lower)
                elif prefix in item_lower:
                    return (1, item_lower)
                else:
                    return (2, item_lower)

            formatted.sort(key=sort_key)
        else:
            formatted.sort(key=str.lower)

        # Truncate if too many results
        if len(formatted) > max_completions:
            remaining = len(formatted) - max_completions
            formatted = formatted[:max_completions] + [
                f"{remaining} more, keep typing to filter further"
            ]
        completions_text = "\n".join(formatted)

        suggestions = []
        for item in completions[:max_completions]:
            entry = {"label": item.get("label")}
            for field in ("detail", "kind", "sortText"):
                if field in item and item[field] is not None:
                    entry[field] = item[field]
            suggestions.append(entry)
        if len(completions) > max_completions:
            suggestions.append(
                {
                    "label": "additional",
                    "detail": f"{len(completions) - max_completions} more",
                }
            )

        payload = {
            "file": rel_path,
            "position": {"line": line, "column": column},
            "prefix": prefix,
            "suggestions": suggestions,
            "line": format_line(content, line, None),
            "line_with_cursor": f_line,
        }

        return ok_response(payload, legacy_text=f"Completions at:\n{f_line}\n{completions_text}")


@mcp.tool("lean_declaration_file")
def declaration_file(ctx: Context, file_path: str, symbol: str) -> Any:
    """Get the file contents where a symbol/lemma/class/structure is declared.

    Note:
        Symbol must be present in the file! Add if necessary!
        Lean files can be large, use `lean_hover_info` before this tool.

    Args:
        file_path (str): Abs path to Lean file
        symbol (str): Symbol to look up the declaration for. Case sensitive!

    Returns:
        str: File contents or error msg
    """
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        message = "Invalid Lean file path: Unable to start LSP server or load file"
        return error_response(
            message, data={"file_path": file_path, "symbol": symbol}
        )
    with client_session(ctx) as client:
        if client is None:
            return error_response(
                "Lean client is not available. Run another tool to initialize the project first.",
                data={"file": rel_path, "symbol": symbol},
                code=ERROR_CLIENT_NOT_READY,
            )

        orig_file_content = update_file(ctx, rel_path)

        # Find the first occurence of the symbol (line and column) in the file,
        position = find_start_position(orig_file_content, symbol)
        if not position:
            message = (
                f"Symbol `{symbol}` (case sensitive) not found in file `{rel_path}`. Add it first,"
                " then try again."
            )
            return error_response(
                f"Symbol `{symbol}` not found",
                data={"file": rel_path, "symbol": symbol},
                legacy_text=message,
            )

        declaration = client.get_declarations(
            rel_path, position["line"], position["column"]
        )

        if len(declaration) == 0:
            message = f"No declaration available for `{symbol}`."
            return error_response(
                message,
                data={"file": rel_path, "symbol": symbol},
                legacy_text=message,
            )

        # Load the declaration file
        declaration = declaration[0]
        uri = declaration.get("targetUri")
        if not uri:
            uri = declaration.get("uri")

        abs_path = client._uri_to_abs(uri)
        if not os.path.exists(abs_path):
            message = f"Could not open declaration file `{abs_path}` for `{symbol}`."
            return error_response(
                message,
                data={"symbol": symbol, "path": abs_path},
                legacy_text=message,
            )

        file_content = get_file_contents(abs_path)

        payload = {
            "symbol": symbol,
            "origin_file": rel_path,
            "declaration": {
                "uri": uri,
                "path": abs_path,
                "contents": file_content,
            },
        }
        legacy = f"Declaration of `{symbol}`:\n{file_content}"
        return ok_response(payload, legacy_text=legacy)


@mcp.tool("lean_multi_attempt")
def multi_attempt(
    ctx: Context, file_path: str, line: int, snippets: List[str]
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
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        message = "Invalid Lean file path: Unable to start LSP server or load file"
        return error_response(
            message, data={"file_path": file_path, "line": line, "snippets": snippets}
        )
    with client_session(ctx) as client:
        if client is None:
            return error_response(
                "Lean client is not available. Run another tool to initialize the project first.",
                data={"file": rel_path, "line": line},
                code=ERROR_CLIENT_NOT_READY,
            )

        update_file(ctx, rel_path)

        try:
            client.open_file(rel_path)

            results = []
            legacy_texts: List[str] = []
            for snippet in snippets:
                payload = snippet if snippet.endswith("\n") else f"{snippet}\n"
                change = DocumentContentChange(
                    payload,
                    [line - 1, 0],
                    [line, 0],
                )
                diag = client.update_file(rel_path, [change])
                formatted_diag = "\n".join(
                    format_diagnostics(diag, select_line=line - 1)
                )
                goal = client.get_goal(rel_path, line - 1, len(snippet))
                formatted_goal = format_goal(goal, "Missing goal")
                results.append(
                    {
                        "snippet": snippet,
                        "goal": goal_to_payload(goal),
                        "diagnostics": diagnostics_to_entries(
                            diag, select_line=line - 1
                        ),
                    }
                )
                legacy_texts.append(f"{snippet}:\n {formatted_goal}\n\n{formatted_diag}")

            legacy = legacy_texts
            payload = {
                "file": rel_path,
                "line": line,
                "attempts": [
                    {
                        "snippet": entry["snippet"],
                        "goal": entry["goal"],
                        "diagnostics": entry["diagnostics"],
                    }
                    for entry in results
                ],
            }
            return ok_response(payload, legacy_text=legacy)
        finally:
            try:
                client.close_files([rel_path])
            except Exception as exc:  # pragma: no cover - close failures only logged
                logger.warning(
                    "Failed to close `%s` after multi_attempt: %s", rel_path, exc
                )


@mcp.tool("lean_run_code")
def run_code(ctx: Context, code: str) -> Any:
    """Run a complete, self-contained code snippet and return diagnostics.

    Has to include all imports and definitions!
    Only use for testing outside open files! Keep the user in the loop by editing files instead.

    Args:
        code (str): Code snippet

    Returns:
        List[str] | str: Diagnostics msgs or error msg
    """
    lean_project_path = ctx.request_context.lifespan_context.lean_project_path
    if lean_project_path is None:
        message = (
            "No valid Lean project path found. Run another tool (e.g. `lean_diagnostic_messages`) first to set it up or set the LEAN_PROJECT_PATH environment variable."
        )
        return error_response(message, data={"code": code})

    rel_path = f"_mcp_snippet_{uuid.uuid4().hex}.lean"
    abs_path = os.path.join(lean_project_path, rel_path)

    try:
        with open(abs_path, "w") as f:
            f.write(code)
    except Exception as e:
        message = f"Error writing code snippet to file `{abs_path}`:\n{str(e)}"
        return error_response(
            "Error writing code snippet",
            data={"path": abs_path, "error": str(e)},
            legacy_text=message,
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
        return error_response(
            "Error starting Lean client",
            data={"path": abs_path, "error": str(e)},
            legacy_text=message,
        )

    diagnostics_payload: List[Dict[str, Any]] | None = None
    legacy_diagnostics: List[str] | str | None = None
    close_error: str | None = None
    remove_error: str | None = None

    with client_session(ctx) as client:
        if client is None:
            try:
                os.remove(abs_path)
            except Exception:
                pass
            return error_response(
                "Lean client is not available. Run another tool to initialize the project first.",
                data={"path": abs_path},
                code=ERROR_CLIENT_NOT_READY,
            )

        try:
            client.open_file(rel_path)
            raw_diagnostics = client.get_diagnostics(rel_path)
            diagnostics_payload = diagnostics_to_entries(raw_diagnostics)
            legacy_diagnostics = format_diagnostics(raw_diagnostics)
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
        return error_response(
            f"Error removing temporary file `{abs_path}`",
            data={"path": abs_path, "error": remove_error},
            legacy_text=f"Error removing temporary file `{abs_path}`:\n{remove_error}",
        )
    if close_error:
        return error_response(
            f"Error closing temporary Lean document `{rel_path}`",
            data={"path": abs_path, "error": close_error},
            legacy_text=f"Error closing temporary Lean document `{rel_path}`:\n{close_error}",
        )

    payload = {
        "snippet_path": rel_path,
        "diagnostics": diagnostics_payload,
    }
    legacy = (
        legacy_diagnostics
        if diagnostics_payload
        else "No diagnostics found for the code snippet (compiled successfully)."
    )
    return ok_response(payload, legacy_text=legacy)


@mcp.tool("lean_tool_spec")
def tool_spec(ctx: Context) -> Any:
    spec = build_tool_spec()
    return ok_response(
        spec,
        meta={"tool_spec_version": TOOL_SPEC_VERSION},
        legacy_text=spec,
    )


@mcp.tool("lean_leansearch")
@rate_limited("leansearch", max_requests=3, per_seconds=30)
def leansearch(ctx: Context, query: str, num_results: int = 5) -> Any:
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
            message = "No results found."
            return error_response(
                message,
                data={"query": query, "num_results": num_results},
                meta={"source": "leansearch"},
                legacy_text=message,
            )
        results = results[0][:num_results]
        results = [r["result"] for r in results]

        for result in results:
            result.pop("docstring")
            result["module_name"] = ".".join(result["module_name"])
            result["name"] = ".".join(result["name"])

        payload = {
            "query": query,
            "results": results,
            "count": len(results),
        }
        return ok_response(payload, meta={"source": "leansearch"}, legacy_text=results)
    except Exception as e:
        message = f"leansearch error:\n{str(e)}"
        return error_response(
            "leansearch error",
            data={"error": str(e), "query": query},
            meta={"source": "leansearch"},
            legacy_text=message,
        )


@mcp.tool("lean_loogle")
@rate_limited("loogle", max_requests=3, per_seconds=30)
def loogle(ctx: Context, query: str, num_results: int = 8) -> Any:
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
    try:
        req = urllib.request.Request(
            f"https://loogle.lean-lang.org/json?q={urllib.parse.quote(query)}",
            headers={"User-Agent": "lean-lsp-mcp/0.1"},
            method="GET",
        )

        with urllib.request.urlopen(req, timeout=20) as response:
            results = json.loads(response.read().decode("utf-8"))

        if "hits" not in results:
            message = "No results found."
            return error_response(
                message,
                data={"query": query, "num_results": num_results},
                meta={"source": "loogle"},
                legacy_text=message,
            )

        results = results["hits"][:num_results]
        for result in results:
            result.pop("doc")
        payload = {
            "query": query,
            "results": results,
            "count": len(results),
        }
        return ok_response(payload, meta={"source": "loogle"}, legacy_text=results)
    except Exception as e:
        message = f"loogle error:\n{str(e)}"
        return error_response(
            "loogle error",
            data={"error": str(e), "query": query},
            meta={"source": "loogle"},
            legacy_text=message,
        )


@mcp.tool("lean_state_search")
@rate_limited("lean_state_search", max_requests=3, per_seconds=30)
def state_search(
    ctx: Context, file_path: str, line: int, column: int, num_results: int = 5
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
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        message = "Invalid Lean file path: Unable to start LSP server or load file"
        return error_response(
            message,
            data={"file_path": file_path, "line": line, "column": column},
            meta={"source": "lean_state_search"},
        )

    with client_session(ctx) as client:
        if client is None:
            return error_response(
                "Lean client is not available. Run another tool to initialize the project first.",
                data={"file": rel_path, "line": line, "column": column},
                meta={"source": "lean_state_search"},
                code=ERROR_CLIENT_NOT_READY,
            )

        file_contents = update_file(ctx, rel_path)
        goal_state = client.get_goal(rel_path, line - 1, column - 1)

    f_line = format_line(file_contents, line, column)
    if not goal_state or not goal_state.get("goals"):
        message = f"No goals found:\n{f_line}\nTry elsewhere?"
        return error_response(
            "No goals found",
            data={"file": rel_path, "line": line, "column": column},
            meta={"source": "lean_state_search"},
            legacy_text=message,
        )

    goal = urllib.parse.quote(goal_state["goals"][0])

    try:
        url = os.getenv("LEAN_STATE_SEARCH_URL", "https://premise-search.com")
        req = urllib.request.Request(
            f"{url}/api/search?query={goal}&results={num_results}&rev=v4.17.0-rc1",
            headers={"User-Agent": "lean-lsp-mcp/0.1"},
            method="GET",
        )

        with urllib.request.urlopen(req, timeout=20) as response:
            results = json.loads(response.read().decode("utf-8"))

        for result in results:
            result.pop("rev")
        payload = {
            "file": rel_path,
            "position": {"line": line, "column": column},
            "query": goal,
            "results": results,
            "count": len(results),
        }
        legacy = [f"Results for line:\n{f_line}"] + results
        return ok_response(
            payload,
            meta={"source": "lean_state_search", "endpoint": url},
            legacy_text=legacy,
        )
    except Exception as e:
        message = f"lean state search error:\n{str(e)}"
        return error_response(
            "lean state search error",
            data={"error": str(e), "file": rel_path, "line": line, "column": column},
            meta={"source": "lean_state_search"},
            legacy_text=message,
        )


@mcp.tool("lean_hammer_premise")
@rate_limited("hammer_premise", max_requests=3, per_seconds=30)
def hammer_premise(
    ctx: Context, file_path: str, line: int, column: int, num_results: int = 32
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
    rel_path = setup_client_for_file(ctx, file_path)
    if not rel_path:
        message = "Invalid Lean file path: Unable to start LSP server or load file"
        return error_response(
            message,
            data={"file_path": file_path, "line": line, "column": column},
            meta={"source": "lean_hammer_premise"},
        )

    with client_session(ctx) as client:
        if client is None:
            return error_response(
                "Lean client is not available. Run another tool to initialize the project first.",
                data={"file": rel_path, "line": line, "column": column},
                meta={"source": "lean_hammer_premise"},
                code=ERROR_CLIENT_NOT_READY,
            )

        file_contents = update_file(ctx, rel_path)
        goal_state = client.get_goal(rel_path, line - 1, column - 1)

    f_line = format_line(file_contents, line, column)
    if not goal_state or not goal_state.get("goals"):
        message = f"No goals found:\n{f_line}\nTry elsewhere?"
        return error_response(
            "No goals found",
            data={"file": rel_path, "line": line, "column": column},
            meta={"source": "lean_hammer_premise"},
            legacy_text=message,
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
        payload = {
            "file": rel_path,
            "position": {"line": line, "column": column},
            "query": data["state"],
            "results": results,
            "count": len(results),
        }
        legacy = [f"Results for line:\n{f_line}"] + results
        return ok_response(
            payload,
            meta={"source": "lean_hammer_premise", "endpoint": url},
            legacy_text=legacy,
        )
    except Exception as e:
        message = f"lean hammer premise error:\n{str(e)}"
        return error_response(
            "lean hammer premise error",
            data={"error": str(e), "file": rel_path, "line": line, "column": column},
            meta={"source": "lean_hammer_premise"},
            legacy_text=message,
        )


if __name__ == "__main__":
    mcp.run()
