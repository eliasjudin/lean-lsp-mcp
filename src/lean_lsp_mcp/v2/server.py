from __future__ import annotations

import asyncio
import functools
import logging.config
import os
import re
import ssl
import time
import urllib
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import certifi
import orjson
from leanclient import DocumentContentChange, LeanLSPClient
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase
from mcp.server.fastmcp.utilities.logging import configure_logging, get_logger
from mcp.types import ToolAnnotations
from pydantic import ConfigDict, Field

from lean_lsp_mcp.client_utils import startup_client
from lean_lsp_mcp.file_utils import get_file_contents, get_relative_file_path
from lean_lsp_mcp.loogle import LoogleManager, loogle_remote
from lean_lsp_mcp.models import (
    AttemptResult,
    BuildResult,
    CompletionItem,
    CompletionsResult,
    DeclarationInfo,
    DiagnosticMessage,
    DiagnosticsResult,
    FileOutline,
    GoalState,
    HoverInfo,
    LeanFinderResult,
    LeanFinderResults,
    LeanSearchResult,
    LeanSearchResults,
    LocalSearchResult,
    LocalSearchResults,
    LoogleResult,
    LoogleResults,
    MultiAttemptResult,
    PremiseResult,
    PremiseResults,
    ProofProfileResult,
    RunResult,
    StateSearchResult,
    StateSearchResults,
    TermGoalState,
)
from lean_lsp_mcp.outline_utils import generate_outline_data
from lean_lsp_mcp.repl import Repl, repl_enabled
from lean_lsp_mcp.search_utils import check_ripgrep_status, lean_local_search
from lean_lsp_mcp.utils import (
    COMPLETION_KIND,
    LeanToolError,
    OutputCapture,
    check_lsp_response,
    extract_failed_dependency_paths,
    extract_goals_list,
    extract_range,
    filter_diagnostics_by_position,
    find_start_position,
    get_declaration_range,
    is_build_stderr,
)
from lean_lsp_mcp.v2.auth import auth_settings_and_verifier
from lean_lsp_mcp.v2.pathing import PathResolutionError, resolve_workspace_path, to_workspace_relative
from lean_lsp_mcp.v2.profiles import ServerProfile, external_tools_enabled, get_server_profile, write_tools_enabled
from lean_lsp_mcp.v2.search_fetch import declaration_text_for_id, search_payload_from_local_results

INSTRUCTIONS_V2 = """## Lean LSP MCP v2
- Remote-first server. Use project-relative `path` only.
- Single workspace root per process.
- Use `search`/`fetch` for OpenAI MCP-compatible document retrieval.
- Read tools are always available; write tools depend on server profile.
"""

# Enforce strict JSON argument validation for all tools (no extra keys).
ArgModelBase.model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
ArgModelBase.model_rebuild(force=True)

# LSP Diagnostic severity: 1=error, 2=warning, 3=info, 4=hint
DIAGNOSTIC_SEVERITY: dict[int, str] = {1: "error", 2: "warning", 3: "info", 4: "hint"}


async def _urlopen_json(req: urllib.request.Request, timeout: float):
    """Run urllib.request.urlopen in a worker thread to avoid blocking the event loop."""
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    def _do_request():
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx) as response:
            return orjson.loads(response.read())

    return await asyncio.to_thread(_do_request)


async def _urlopen_json_retry(
    req: urllib.request.Request,
    *,
    timeout: float = 10.0,
    retries: int = 2,
    backoff_seconds: float = 0.75,
):
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await _urlopen_json(req, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == retries:
                break
            await asyncio.sleep(backoff_seconds * (2**attempt))
    raise LeanToolError(f"External request failed after retries: {last_exc}")


async def _safe_report_progress(
    ctx: Context, *, progress: int, total: int, message: str
) -> None:
    try:
        await ctx.report_progress(progress=progress, total=total, message=message)
    except Exception:
        return


_LOG_FILE_CONFIG = os.environ.get("LEAN_LOG_FILE_CONFIG", None)
_LOG_LEVEL = os.environ.get("LEAN_LOG_LEVEL", "INFO")
if _LOG_FILE_CONFIG:
    try:
        if _LOG_FILE_CONFIG.endswith((".yaml", ".yml")):
            import yaml

            with open(_LOG_FILE_CONFIG, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            logging.config.dictConfig(cfg)
        elif _LOG_FILE_CONFIG.endswith(".json"):
            with open(_LOG_FILE_CONFIG, "r", encoding="utf-8") as f:
                cfg = orjson.loads(f.read())
            logging.config.dictConfig(cfg)
        else:
            logging.config.fileConfig(_LOG_FILE_CONFIG, disable_existing_loggers=False)
    except Exception as e:
        configure_logging("CRITICAL" if _LOG_LEVEL == "NONE" else _LOG_LEVEL)
        logger = get_logger(__name__)
        logger.warning(
            "Failed to load logging config %s: %s. Falling back to LEAN_LOG_LEVEL.",
            _LOG_FILE_CONFIG,
            e,
        )
else:
    configure_logging("CRITICAL" if _LOG_LEVEL == "NONE" else _LOG_LEVEL)

logger = get_logger(__name__)

_RG_AVAILABLE, _RG_MESSAGE = check_ripgrep_status()
SERVER_PROFILE = get_server_profile()
READ_TOOL_NAMES: tuple[str, ...] = (
    "search",
    "fetch",
    "outline",
    "diagnostics",
    "goal",
    "term_goal",
    "hover",
    "completions",
    "declaration",
    "local_search",
    "leansearch",
    "loogle",
    "leanfinder",
    "state_search",
    "hammer_premise",
    "profile_proof",
)
WRITE_TOOL_NAMES: tuple[str, ...] = (
    "build",
    "multi_attempt",
    "run_code",
)


@dataclass
class AppContext:
    workspace_root: Path
    lean_project_path: Path
    client: LeanLSPClient | None
    rate_limit: dict[str, list[int]]
    lean_search_available: bool
    loogle_manager: LoogleManager | None = None
    loogle_local_available: bool = False
    repl: Repl | None = None
    repl_enabled: bool = False
    profile: ServerProfile = ServerProfile.READ


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    loogle_manager: LoogleManager | None = None
    loogle_local_available = False
    repl: Repl | None = None
    repl_on = False

    root_raw = os.environ.get("LEAN_WORKSPACE_ROOT", "").strip()
    if not root_raw:
        raise RuntimeError(
            "LEAN_WORKSPACE_ROOT is required in v2 (single-tenant workspace mode)."
        )

    workspace_root = Path(root_raw).expanduser().resolve()
    if not workspace_root.exists() or not workspace_root.is_dir():
        raise RuntimeError(f"LEAN_WORKSPACE_ROOT does not exist: {workspace_root}")
    if not (workspace_root / "lean-toolchain").exists():
        raise RuntimeError(
            "LEAN_WORKSPACE_ROOT must point to a Lean project root containing lean-toolchain."
        )

    try:
        if os.environ.get("LEAN_LOOGLE_LOCAL", "").lower() in ("1", "true", "yes"):
            logger.info("Local loogle enabled, initializing...")
            loogle_manager = LoogleManager(project_path=workspace_root)
            if loogle_manager.ensure_installed():
                if await loogle_manager.start():
                    loogle_local_available = True
                    logger.info("Local loogle started successfully")
                else:
                    logger.warning("Local loogle failed to start, will use remote API")
            else:
                logger.warning("Local loogle installation failed, will use remote API")

        if repl_enabled():
            from lean_lsp_mcp.repl import find_repl_binary

            repl_bin = find_repl_binary(str(workspace_root))
            if repl_bin:
                logger.info("REPL enabled, using: %s", repl_bin)
                repl = Repl(project_dir=str(workspace_root), repl_path=repl_bin)
                repl_on = True
                logger.info("REPL initialized: timeout=%ds", repl.timeout)
            else:
                logger.warning(
                    "REPL enabled but binary not found. "
                    'Add `require repl from git "https://github.com/leanprover-community/repl"` '
                    "to lakefile and run `lake build repl`. Falling back to LSP."
                )

        context = AppContext(
            workspace_root=workspace_root,
            lean_project_path=workspace_root,
            client=None,
            rate_limit={
                "leansearch": [],
                "loogle": [],
                "leanfinder": [],
                "lean_state_search": [],
                "hammer_premise": [],
            },
            lean_search_available=_RG_AVAILABLE,
            loogle_manager=loogle_manager,
            loogle_local_available=loogle_local_available,
            repl=repl,
            repl_enabled=repl_on,
            profile=SERVER_PROFILE,
        )
        enabled_tools = list(READ_TOOL_NAMES)
        if write_tools_enabled(SERVER_PROFILE):
            enabled_tools.extend(WRITE_TOOL_NAMES)
        logger.info(
            "lean-lsp-mcp v2 started: profile=%s workspace=%s tools=%s",
            SERVER_PROFILE.value,
            workspace_root,
            ",".join(enabled_tools),
        )
        yield context
    finally:
        logger.info("Closing Lean LSP client")
        if context.client:
            context.client.close()
        if loogle_manager:
            await loogle_manager.stop()
        if repl:
            await repl.close()


def _resolve_and_prepare_file(ctx: Context, path: str) -> tuple[Path, str]:
    app_ctx: AppContext = ctx.request_context.lifespan_context
    try:
        abs_path = resolve_workspace_path(app_ctx.workspace_root, path)
    except PathResolutionError as exc:
        raise LeanToolError(str(exc)) from exc

    if not abs_path.exists() or not abs_path.is_file():
        raise LeanToolError(f"Path does not exist: {path}")

    rel = get_relative_file_path(app_ctx.lean_project_path, str(abs_path))
    if not rel:
        raise LeanToolError("File path is outside Lean project root.")

    startup_client(ctx)
    return abs_path, rel


def _enforce_argument_keys(ctx: Context, *, allowed: set[str]) -> None:
    """Reject tool calls that include unexpected top-level argument keys."""
    request = getattr(ctx.request_context, "request", None)
    params = getattr(request, "params", None)
    raw_args = getattr(params, "arguments", None)
    if not isinstance(raw_args, dict):
        return
    unexpected = set(raw_args.keys()) - allowed
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise LeanToolError(f"Unexpected argument(s): {names}")


def rate_limited(category: str, max_requests: int, per_seconds: int):
    def decorator(func):
        def _apply_rate_limit(args, kwargs):
            ctx = kwargs.get("ctx")
            if ctx is None:
                if not args:
                    raise KeyError(
                        "rate_limited wrapper requires ctx as a keyword argument or first positional argument"
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
                return (
                    False,
                    f"Tool limit exceeded: {max_requests} requests per {per_seconds} s. Try again later.",
                )
            rate_limit[category].append(current_time)
            return True, None

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                allowed, msg = _apply_rate_limit(args, kwargs)
                if not allowed:
                    return msg
                return await func(*args, **kwargs)

        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                allowed, msg = _apply_rate_limit(args, kwargs)
                if not allowed:
                    return msg
                return func(*args, **kwargs)

        doc = wrapper.__doc__ or ""
        wrapper.__doc__ = f"Limit: {max_requests}req/{per_seconds}s. {doc}"
        return wrapper

    return decorator


auth_settings, token_verifier = auth_settings_and_verifier()

mcp_kwargs = dict(
    name="Lean LSP v2",
    instructions=INSTRUCTIONS_V2,
    dependencies=["leanclient"],
    lifespan=app_lifespan,
)
if auth_settings and token_verifier:
    mcp_kwargs["auth"] = auth_settings
    mcp_kwargs["token_verifier"] = token_verifier

mcp = FastMCP(**mcp_kwargs)


def _to_diagnostic_messages(diagnostics: list[dict]) -> list[DiagnosticMessage]:
    result = []
    for diag in diagnostics:
        r = diag.get("fullRange", diag.get("range"))
        if r is None:
            continue
        severity_int = diag.get("severity", 1)
        result.append(
            DiagnosticMessage(
                severity=DIAGNOSTIC_SEVERITY.get(severity_int, f"unknown({severity_int})"),
                message=diag.get("message", ""),
                line=r["start"]["line"] + 1,
                column=r["start"]["character"] + 1,
            )
        )
    return result


def _process_diagnostics(diagnostics: list[dict], build_success: bool) -> DiagnosticsResult:
    items = []
    failed_deps: list[str] = []

    for diag in diagnostics:
        r = diag.get("fullRange", diag.get("range"))
        if r is None:
            continue

        severity_int = diag.get("severity", 1)
        message = diag.get("message", "")
        line = r["start"]["line"] + 1
        column = r["start"]["character"] + 1

        if line == 1 and column == 1 and is_build_stderr(message):
            failed_deps = extract_failed_dependency_paths(message)
            continue

        items.append(
            DiagnosticMessage(
                severity=DIAGNOSTIC_SEVERITY.get(severity_int, f"unknown({severity_int})"),
                message=message,
                line=line,
                column=column,
            )
        )

    return DiagnosticsResult(
        success=build_success,
        items=items,
        failed_dependencies=failed_deps,
    )


def _diagnostics_success_from_items(diagnostics: list[dict]) -> bool:
    """Compute success flag for leanclient versions returning plain diagnostic lists."""
    for diag in diagnostics:
        if not isinstance(diag, dict):
            continue
        if diag.get("severity", 1) == 1:
            return False
    return True


@mcp.tool(
    "search",
    annotations=ToolAnnotations(
        title="Search",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def search(
    ctx: Context,
    query: Annotated[str, Field(description="Search query")],
) -> str:
    """OpenAI-compatible MCP search tool returning JSON payload in text content."""
    _enforce_argument_keys(ctx, allowed={"query"})
    app_ctx: AppContext = ctx.request_context.lifespan_context
    if not app_ctx.lean_search_available:
        raise LeanToolError(_RG_MESSAGE)

    raw = lean_local_search(query=query.strip(), limit=20, project_root=app_ctx.workspace_root)
    local = LocalSearchResults(
        items=[
            LocalSearchResult(name=r["name"], kind=r["kind"], file=r["file"])
            for r in raw
        ]
    )
    payload = search_payload_from_local_results(local, app_ctx.workspace_root)
    return payload.model_dump_json()


@mcp.tool(
    "fetch",
    annotations=ToolAnnotations(
        title="Fetch",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def fetch(
    ctx: Context,
    id: Annotated[str, Field(description="Declaration id from `search`")],
) -> str:
    """OpenAI-compatible MCP fetch tool returning JSON payload in text content."""
    _enforce_argument_keys(ctx, allowed={"id"})
    app_ctx: AppContext = ctx.request_context.lifespan_context
    startup_client(ctx)
    client: LeanLSPClient = app_ctx.client
    payload = declaration_text_for_id(
        workspace_root=app_ctx.workspace_root,
        client=client,
        identifier=id,
    )
    return payload.model_dump_json()


def _force_strict_input_schema(*tool_names: str) -> None:
    """Ensure selected tool schemas reject unknown properties."""
    for tool_name in tool_names:
        tool = mcp._tool_manager._tools.get(tool_name)
        if tool is None:
            continue
        params = getattr(tool, "parameters", None)
        if isinstance(params, dict):
            params["additionalProperties"] = False


_force_strict_input_schema("search", "fetch")


@mcp.tool(
    "outline",
    annotations=ToolAnnotations(
        title="File Outline",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def outline(
    ctx: Context,
    path: Annotated[str, Field(description="Workspace-relative path to Lean file")],
) -> FileOutline:
    _, rel_path = _resolve_and_prepare_file(ctx, path)
    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    return generate_outline_data(client, rel_path)


@mcp.tool(
    "diagnostics",
    annotations=ToolAnnotations(
        title="Diagnostics",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def diagnostics(
    ctx: Context,
    path: Annotated[str, Field(description="Workspace-relative path to Lean file")],
    start_line: Annotated[int | None, Field(description="Filter from line", ge=1)] = None,
    end_line: Annotated[int | None, Field(description="Filter to line", ge=1)] = None,
    declaration_name: Annotated[
        str | None, Field(description="Filter to declaration (slow)")
    ] = None,
) -> DiagnosticsResult:
    _, rel_path = _resolve_and_prepare_file(ctx, path)
    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)

    if declaration_name:
        decl_range = get_declaration_range(client, rel_path, declaration_name)
        if decl_range is None:
            raise LeanToolError(f"Declaration '{declaration_name}' not found in file.")
        start_line, end_line = decl_range

    start_line_0 = (start_line - 1) if start_line is not None else None
    end_line_0 = (end_line - 1) if end_line is not None else None

    result = client.get_diagnostics(
        rel_path,
        start_line=start_line_0,
        end_line=end_line_0,
        inactivity_timeout=15.0,
    )
    check_lsp_response(result, "get_diagnostics")

    if isinstance(result, list):
        diagnostics_list = result
        success = _diagnostics_success_from_items(diagnostics_list)
        return _process_diagnostics(diagnostics_list, success)

    # Backward compatibility for leanclient variants that return wrapper objects.
    diagnostics_list = getattr(result, "diagnostics", None)
    if not isinstance(diagnostics_list, list):
        raise LeanToolError("Unexpected diagnostics response shape from Lean client.")
    success_attr = getattr(result, "success", None)
    success = (
        bool(success_attr)
        if isinstance(success_attr, bool)
        else _diagnostics_success_from_items(diagnostics_list)
    )
    return _process_diagnostics(diagnostics_list, success)


@mcp.tool(
    "goal",
    annotations=ToolAnnotations(
        title="Proof Goals",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def goal(
    ctx: Context,
    path: Annotated[str, Field(description="Workspace-relative path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[
        int | None,
        Field(description="Column (1-indexed). Omit for before/after", ge=1),
    ] = None,
) -> GoalState:
    _, rel_path = _resolve_and_prepare_file(ctx, path)
    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    content = client.get_file_content(rel_path)
    lines = content.splitlines()

    if line < 1 or line > len(lines):
        raise LeanToolError(f"Line {line} out of range (file has {len(lines)} lines)")

    line_context = lines[line - 1]

    if column is None:
        column_end = len(line_context)
        column_start = next((i for i, c in enumerate(line_context) if not c.isspace()), 0)
        goal_start = client.get_goal(rel_path, line - 1, column_start)
        check_lsp_response(goal_start, "get_goal", allow_none=True)
        goal_end = client.get_goal(rel_path, line - 1, column_end)
        return GoalState(
            line_context=line_context,
            goals_before=extract_goals_list(goal_start),
            goals_after=extract_goals_list(goal_end),
        )

    goal_result = client.get_goal(rel_path, line - 1, column - 1)
    check_lsp_response(goal_result, "get_goal", allow_none=True)
    return GoalState(line_context=line_context, goals=extract_goals_list(goal_result))


@mcp.tool(
    "term_goal",
    annotations=ToolAnnotations(
        title="Term Goal",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def term_goal(
    ctx: Context,
    path: Annotated[str, Field(description="Workspace-relative path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[int | None, Field(description="Column (defaults to end of line)", ge=1)] = None,
) -> TermGoalState:
    _, rel_path = _resolve_and_prepare_file(ctx, path)
    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    content = client.get_file_content(rel_path)
    lines = content.splitlines()

    if line < 1 or line > len(lines):
        raise LeanToolError(f"Line {line} out of range (file has {len(lines)} lines)")

    line_context = lines[line - 1]
    if column is None:
        column = len(line_context)

    term_goal_result = client.get_term_goal(rel_path, line - 1, column - 1)
    check_lsp_response(term_goal_result, "get_term_goal", allow_none=True)
    expected_type = None
    if term_goal_result is not None:
        rendered = term_goal_result.get("goal")
        if rendered:
            expected_type = rendered.replace("```lean\n", "").replace("\n```", "")

    return TermGoalState(line_context=line_context, expected_type=expected_type)


@mcp.tool(
    "hover",
    annotations=ToolAnnotations(
        title="Hover Info",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def hover(
    ctx: Context,
    path: Annotated[str, Field(description="Workspace-relative path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[int, Field(description="Column at START of identifier", ge=1)],
) -> HoverInfo:
    _, rel_path = _resolve_and_prepare_file(ctx, path)
    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    file_content = client.get_file_content(rel_path)
    hover_info = client.get_hover(rel_path, line - 1, column - 1)
    check_lsp_response(hover_info, "get_hover", allow_none=True)
    if hover_info is None:
        raise LeanToolError(f"No hover information at line {line}, column {column}")

    h_range = hover_info.get("range")
    symbol = extract_range(file_content, h_range) or ""
    info = hover_info["contents"].get("value", "No hover information available.")
    info = info.replace("```lean\n", "").replace("\n```", "").strip()

    diagnostics = client.get_diagnostics(rel_path)
    check_lsp_response(diagnostics, "get_diagnostics")
    filtered = filter_diagnostics_by_position(diagnostics, line - 1, column - 1)

    return HoverInfo(
        symbol=symbol,
        info=info,
        diagnostics=_to_diagnostic_messages(filtered),
    )


@mcp.tool(
    "completions",
    annotations=ToolAnnotations(
        title="Completions",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def completions(
    ctx: Context,
    path: Annotated[str, Field(description="Workspace-relative path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[int, Field(description="Column number (1-indexed)", ge=1)],
    max_completions: Annotated[int, Field(description="Max completions", ge=1)] = 32,
) -> CompletionsResult:
    _, rel_path = _resolve_and_prepare_file(ctx, path)
    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    content = client.get_file_content(rel_path)
    raw_completions = client.get_completions(rel_path, line - 1, column - 1)
    check_lsp_response(raw_completions, "get_completions")

    items: list[CompletionItem] = []
    for c in raw_completions:
        if "label" not in c:
            continue
        kind_int = c.get("kind")
        kind_str = COMPLETION_KIND.get(kind_int) if kind_int else None
        items.append(
            CompletionItem(
                label=c["label"],
                kind=kind_str,
                detail=c.get("detail"),
            )
        )

    if not items:
        return CompletionsResult(items=[])

    lines = content.splitlines()
    prefix = ""
    if 0 < line <= len(lines):
        text_before_cursor = lines[line - 1][: column - 1] if column > 0 else ""
        if not text_before_cursor.endswith("."):
            prefix = re.split(r"[\s()\[\]{},:;.]+", text_before_cursor)[-1].lower()

    if prefix:

        def sort_key(item: CompletionItem):
            label_lower = item.label.lower()
            if label_lower.startswith(prefix):
                return (0, label_lower)
            if prefix in label_lower:
                return (1, label_lower)
            return (2, label_lower)

        items.sort(key=sort_key)
    else:
        items.sort(key=lambda x: x.label.lower())

    return CompletionsResult(items=items[:max_completions])


@mcp.tool(
    "declaration",
    annotations=ToolAnnotations(
        title="Declaration Source",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def declaration(
    ctx: Context,
    path: Annotated[str, Field(description="Workspace-relative path to Lean file")],
    symbol: Annotated[str, Field(description="Symbol (case sensitive, must be in file)")],
) -> DeclarationInfo:
    _, rel_path = _resolve_and_prepare_file(ctx, path)
    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    orig_file_content = client.get_file_content(rel_path)

    position = find_start_position(orig_file_content, symbol)
    if not position:
        raise LeanToolError(
            f"Symbol `{symbol}` (case sensitive) not found in file. Add it first."
        )

    declaration = client.get_declarations(rel_path, position["line"], position["column"])

    if len(declaration) == 0:
        raise LeanToolError(f"No declaration available for `{symbol}`.")

    decl = declaration[0]
    uri = decl.get("targetUri") or decl.get("uri")

    abs_path = Path(client._uri_to_abs(uri)).resolve()
    if not abs_path.exists():
        raise LeanToolError(f"Could not open declaration file `{abs_path}` for `{symbol}`.")

    try:
        rel_path = to_workspace_relative(ctx.request_context.lifespan_context.workspace_root, abs_path)
    except PathResolutionError as exc:
        raise LeanToolError(str(exc)) from exc

    file_content = get_file_contents(str(abs_path))
    return DeclarationInfo(path=rel_path, content=file_content)


async def _multi_attempt_repl(
    ctx: Context,
    abs_path: Path,
    rel_path: str,
    line: int,
    snippets: list[str],
) -> MultiAttemptResult | None:
    app_ctx: AppContext = ctx.request_context.lifespan_context
    if not app_ctx.repl_enabled or not app_ctx.repl:
        return None

    try:
        content = get_file_contents(str(abs_path))
        lines = content.splitlines()
        if line > len(lines):
            return None

        base_code = "\n".join(lines[: line - 1])
        repl_results = await app_ctx.repl.run_snippets(base_code, snippets)

        results = []
        for snippet, pr in zip(snippets, repl_results):
            diagnostics = [
                DiagnosticMessage(
                    severity=m.get("severity", "info"),
                    message=m.get("data", ""),
                    line=m.get("pos", {}).get("line", 0),
                    column=m.get("pos", {}).get("column", 0),
                )
                for m in (pr.messages or [])
            ]
            if pr.error:
                diagnostics.append(
                    DiagnosticMessage(
                        severity="error", message=pr.error, line=0, column=0
                    )
                )
            results.append(
                AttemptResult(
                    snippet=snippet.rstrip("\n"),
                    goals=pr.goals or [],
                    diagnostics=diagnostics,
                )
            )
        return MultiAttemptResult(items=results)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"REPL multi_attempt failed: {e}")
        return None


def _multi_attempt_lsp(
    ctx: Context,
    abs_path: Path,
    rel_path: str,
    line: int,
    snippets: list[str],
) -> MultiAttemptResult:
    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    original_content = get_file_contents(str(abs_path))

    try:
        results: list[AttemptResult] = []
        for snippet in snippets:
            snippet_str = snippet.rstrip("\n")
            payload = f"{snippet_str}\n"
            change = DocumentContentChange(
                payload,
                [line - 1, 0],
                [line, 0],
            )
            client.update_file(rel_path, [change])
            diag = client.get_diagnostics(rel_path)
            check_lsp_response(diag, "get_diagnostics")
            filtered_diag = filter_diagnostics_by_position(diag, line - 1, None)
            goal_result = client.get_goal(rel_path, line - 1, len(snippet_str))
            goals = extract_goals_list(goal_result)
            results.append(
                AttemptResult(
                    snippet=snippet_str,
                    goals=goals,
                    diagnostics=_to_diagnostic_messages(filtered_diag),
                )
            )

        return MultiAttemptResult(items=results)
    finally:
        try:
            client.update_file_content(rel_path, original_content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to restore `%s` after multi_attempt: %s", rel_path, exc)


@mcp.tool(
    "local_search",
    annotations=ToolAnnotations(
        title="Local Search",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def local_search(
    ctx: Context,
    query: Annotated[str, Field(description="Declaration name or prefix")],
    limit: Annotated[int, Field(description="Max matches", ge=1)] = 10,
) -> LocalSearchResults:
    if not _RG_AVAILABLE:
        raise LeanToolError(_RG_MESSAGE)

    app_ctx: AppContext = ctx.request_context.lifespan_context
    try:
        raw_results = await asyncio.to_thread(
            lean_local_search,
            query=query.strip(),
            limit=limit,
            project_root=app_ctx.workspace_root,
        )
    except RuntimeError as exc:
        raise LeanToolError(f"Search failed: {exc}") from exc

    items = [
        LocalSearchResult(name=r["name"], kind=r["kind"], file=r["file"])
        for r in raw_results
    ]
    return LocalSearchResults(items=items)


@mcp.tool(
    "leansearch",
    annotations=ToolAnnotations(
        title="LeanSearch",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@rate_limited("leansearch", max_requests=3, per_seconds=30)
async def leansearch(
    ctx: Context,
    query: Annotated[str, Field(description="Natural language or Lean term query")],
    num_results: Annotated[int, Field(description="Max results", ge=1)] = 5,
) -> LeanSearchResults:
    if not external_tools_enabled("leansearch"):
        raise LeanToolError("leansearch is disabled by server configuration.")

    headers = {"User-Agent": "lean-lsp-mcp-v2/1.0", "Content-Type": "application/json"}
    payload = orjson.dumps({"num_results": str(num_results), "query": [query]})
    req = urllib.request.Request(
        "https://leansearch.net/search",
        data=payload,
        headers=headers,
        method="POST",
    )

    await _safe_report_progress(ctx, progress=1, total=10, message="Awaiting leansearch.net")
    results = await _urlopen_json_retry(req, timeout=10)

    if not results or not results[0]:
        return LeanSearchResults(items=[])

    raw_results = [r["result"] for r in results[0][:num_results]]
    items = [
        LeanSearchResult(
            name=".".join(r["name"]),
            module_name=".".join(r["module_name"]),
            kind=r.get("kind"),
            type=r.get("type"),
        )
        for r in raw_results
    ]
    return LeanSearchResults(items=items)


@mcp.tool(
    "loogle",
    annotations=ToolAnnotations(
        title="Loogle",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def loogle(
    ctx: Context,
    query: Annotated[str, Field(description="Type pattern, constant, or name substring")],
    num_results: Annotated[int, Field(description="Max results", ge=1)] = 8,
) -> LoogleResults:
    if not external_tools_enabled("loogle"):
        raise LeanToolError("loogle is disabled by server configuration.")

    app_ctx: AppContext = ctx.request_context.lifespan_context

    if app_ctx.loogle_local_available and app_ctx.loogle_manager:
        if app_ctx.lean_project_path != app_ctx.loogle_manager.project_path:
            if app_ctx.loogle_manager.set_project_path(app_ctx.lean_project_path):
                await app_ctx.loogle_manager.stop()
        try:
            results = await app_ctx.loogle_manager.query(query, num_results)
            items = [
                LoogleResult(
                    name=r.get("name", ""),
                    type=r.get("type", ""),
                    module=r.get("module", ""),
                )
                for r in results
            ]
            return LoogleResults(items=items)
        except Exception as e:  # noqa: BLE001
            logger.warning("Local loogle failed: %s, falling back to remote", e)

    rate_limit = app_ctx.rate_limit["loogle"]
    now = int(time.time())
    rate_limit[:] = [t for t in rate_limit if now - t < 30]
    if len(rate_limit) >= 3:
        raise LeanToolError(
            "Rate limit exceeded: 3 requests per 30s. Use --loogle-local to avoid limits."
        )
    rate_limit.append(now)

    await _safe_report_progress(ctx, progress=1, total=10, message="Awaiting loogle")
    result = await asyncio.to_thread(loogle_remote, query, num_results)
    if isinstance(result, str):
        raise LeanToolError(result)
    return LoogleResults(items=result)


@mcp.tool(
    "leanfinder",
    annotations=ToolAnnotations(
        title="Lean Finder",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@rate_limited("leanfinder", max_requests=10, per_seconds=30)
async def leanfinder(
    ctx: Context,
    query: Annotated[str, Field(description="Mathematical concept or proof state")],
    num_results: Annotated[int, Field(description="Max results", ge=1)] = 5,
) -> LeanFinderResults:
    if not external_tools_enabled("leanfinder"):
        raise LeanToolError("leanfinder is disabled by server configuration.")

    headers = {"User-Agent": "lean-lsp-mcp-v2/1.0", "Content-Type": "application/json"}
    request_url = "https://bxrituxuhpc70w8w.us-east-1.aws.endpoints.huggingface.cloud"
    payload = orjson.dumps({"inputs": query, "top_k": int(num_results)})
    req = urllib.request.Request(request_url, data=payload, headers=headers, method="POST")

    await _safe_report_progress(ctx, progress=1, total=10, message="Awaiting Lean Finder")
    data = await _urlopen_json_retry(req, timeout=10)
    results: list[LeanFinderResult] = []

    for result in data.get("results", []):
        if "https://leanprover-community.github.io/mathlib4_docs" not in result.get("url", ""):
            continue
        match = re.search(r"pattern=(.*?)#doc", result["url"])
        if match:
            results.append(
                LeanFinderResult(
                    full_name=match.group(1),
                    formal_statement=result.get("formal_statement", ""),
                    informal_statement=result.get("informal_statement", ""),
                )
            )

    return LeanFinderResults(items=results)


@mcp.tool(
    "state_search",
    annotations=ToolAnnotations(
        title="State Search",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@rate_limited("lean_state_search", max_requests=3, per_seconds=30)
async def state_search(
    ctx: Context,
    path: Annotated[str, Field(description="Workspace-relative path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[int, Field(description="Column number (1-indexed)", ge=1)],
    num_results: Annotated[int, Field(description="Max results", ge=1)] = 5,
) -> StateSearchResults:
    if not external_tools_enabled("state_search"):
        raise LeanToolError("state_search is disabled by server configuration.")

    _, rel_path = _resolve_and_prepare_file(ctx, path)
    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    goal = client.get_goal(rel_path, line - 1, column - 1)

    if not goal or not goal.get("goals"):
        raise LeanToolError(
            f"No goals found at line {line}, column {column}. Try a different position."
        )

    goal_str = urllib.parse.quote(goal["goals"][0])
    url = os.getenv("LEAN_STATE_SEARCH_URL", "https://premise-search.com")
    req = urllib.request.Request(
        f"{url}/api/search?query={goal_str}&results={num_results}&rev=v4.22.0",
        headers={"User-Agent": "lean-lsp-mcp-v2/1.0"},
        method="GET",
    )

    await _safe_report_progress(ctx, progress=1, total=10, message=f"Awaiting {url}")
    results = await _urlopen_json_retry(req, timeout=10)
    items = [StateSearchResult(name=r["name"]) for r in results]
    return StateSearchResults(items=items)


@mcp.tool(
    "hammer_premise",
    annotations=ToolAnnotations(
        title="Hammer Premises",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@rate_limited("hammer_premise", max_requests=3, per_seconds=30)
async def hammer_premise(
    ctx: Context,
    path: Annotated[str, Field(description="Workspace-relative path to Lean file")],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[int, Field(description="Column number (1-indexed)", ge=1)],
    num_results: Annotated[int, Field(description="Max results", ge=1)] = 32,
) -> PremiseResults:
    if not external_tools_enabled("hammer_premise"):
        raise LeanToolError("hammer_premise is disabled by server configuration.")

    _, rel_path = _resolve_and_prepare_file(ctx, path)
    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    goal = client.get_goal(rel_path, line - 1, column - 1)

    if not goal or not goal.get("goals"):
        raise LeanToolError(
            f"No goals found at line {line}, column {column}. Try a different position."
        )

    data = {"state": goal["goals"][0], "new_premises": [], "k": num_results}
    url = os.getenv("LEAN_HAMMER_URL", "http://leanpremise.net")
    req = urllib.request.Request(
        url + "/retrieve",
        headers={"User-Agent": "lean-lsp-mcp-v2/1.0", "Content-Type": "application/json"},
        method="POST",
        data=orjson.dumps(data),
    )

    await _safe_report_progress(ctx, progress=1, total=10, message=f"Awaiting {url}")
    results = await _urlopen_json_retry(req, timeout=10)

    items = [PremiseResult(name=r["name"]) for r in results]
    return PremiseResults(items=items)


@mcp.tool(
    "profile_proof",
    annotations=ToolAnnotations(
        title="Profile Proof",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def profile_proof(
    ctx: Context,
    path: Annotated[str, Field(description="Workspace-relative path to Lean file")],
    line: Annotated[int, Field(description="Line where theorem starts (1-indexed)", ge=1)],
    top_n: Annotated[int, Field(description="Number of slowest lines to return", ge=1)] = 5,
    timeout: Annotated[float, Field(description="Max seconds to wait", ge=1)] = 60.0,
) -> ProofProfileResult:
    from lean_lsp_mcp.profile_utils import profile_theorem

    abs_path, _ = _resolve_and_prepare_file(ctx, path)
    app_ctx: AppContext = ctx.request_context.lifespan_context

    try:
        return await profile_theorem(
            file_path=abs_path,
            theorem_line=line,
            project_path=app_ctx.lean_project_path,
            timeout=timeout,
            top_n=top_n,
        )
    except (ValueError, TimeoutError) as e:
        raise LeanToolError(str(e)) from e


if write_tools_enabled(SERVER_PROFILE):

    @mcp.tool(
        "build",
        annotations=ToolAnnotations(
            title="Build Project",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def build(
        ctx: Context,
        clean: Annotated[bool, Field(description="Run lake clean first (slow)")] = False,
        output_lines: Annotated[int, Field(description="Return last N lines of build log (0=none)")] = 20,
    ) -> BuildResult:
        app_ctx: AppContext = ctx.request_context.lifespan_context
        lean_project_path_obj = app_ctx.lean_project_path

        log_lines: list[str] = []
        errors: list[str] = []

        try:
            client: LeanLSPClient | None = app_ctx.client
            if client:
                app_ctx.client = None
                client.close()

            if clean:
                await ctx.report_progress(progress=1, total=16, message="Running `lake clean`")
                clean_proc = await asyncio.create_subprocess_exec("lake", "clean", cwd=lean_project_path_obj)
                await clean_proc.wait()

            await ctx.report_progress(progress=2, total=16, message="Running `lake exe cache get`")
            cache_proc = await asyncio.create_subprocess_exec(
                "lake", "exe", "cache", "get", cwd=lean_project_path_obj
            )
            await cache_proc.wait()

            process = await asyncio.create_subprocess_exec(
                "lake",
                "build",
                "--verbose",
                cwd=lean_project_path_obj,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            while line := await process.stdout.readline():
                line_str = line.decode("utf-8", errors="replace").rstrip()

                if line_str.startswith("trace:") or "LEAN_PATH=" in line_str:
                    continue

                log_lines.append(line_str)
                if "error" in line_str.lower():
                    errors.append(line_str)

                if m := re.search(r"\[(\d+)/(\d+)\]\s*(.+?)(?:\s+\(\d+\.?\d*[ms]+\))?$", line_str):
                    await ctx.report_progress(
                        progress=int(m.group(1)),
                        total=int(m.group(2)),
                        message=m.group(3) or "Building",
                    )

            await process.wait()

            if process.returncode != 0:
                return BuildResult(
                    success=False,
                    output="\n".join(log_lines[-output_lines:]) if output_lines else "",
                    errors=errors or [f"Build failed with return code {process.returncode}"],
                )

            with OutputCapture():
                client = LeanLSPClient(
                    lean_project_path_obj, initial_build=False, prevent_cache_get=True
                )

            app_ctx.client = client
            return BuildResult(
                success=True,
                output="\n".join(log_lines[-output_lines:]) if output_lines else "",
                errors=[],
            )

        except Exception as e:  # noqa: BLE001
            return BuildResult(
                success=False,
                output="\n".join(log_lines[-output_lines:]) if output_lines else "",
                errors=[str(e)],
            )


    @mcp.tool(
        "multi_attempt",
        annotations=ToolAnnotations(
            title="Multi-Attempt",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def multi_attempt(
        ctx: Context,
        path: Annotated[str, Field(description="Workspace-relative path to Lean file")],
        line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
        snippets: Annotated[list[str], Field(description="Tactics to try (3+ recommended)")],
    ) -> MultiAttemptResult:
        abs_path, rel_path = _resolve_and_prepare_file(ctx, path)

        result = await _multi_attempt_repl(ctx, abs_path, rel_path, line, snippets)
        if result is not None:
            return result

        return _multi_attempt_lsp(ctx, abs_path, rel_path, line, snippets)


    @mcp.tool(
        "run_code",
        annotations=ToolAnnotations(
            title="Run Code",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    def run_code(
        ctx: Context,
        code: Annotated[str, Field(description="Self-contained Lean code with imports")],
    ) -> RunResult:
        app_ctx: AppContext = ctx.request_context.lifespan_context
        lean_project_path = app_ctx.lean_project_path

        rel_path = f"_mcp_snippet_{uuid.uuid4().hex}.lean"
        abs_path = lean_project_path / rel_path

        try:
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:  # noqa: BLE001
            raise LeanToolError(f"Error writing code snippet: {e}") from e

        client: LeanLSPClient | None = app_ctx.client
        raw_diagnostics: list[dict] = []
        opened_file = False

        try:
            if client is None:
                startup_client(ctx)
                client = app_ctx.client
                if client is None:
                    raise LeanToolError("Failed to initialize Lean client for run_code.")

            client.open_file(rel_path)
            opened_file = True
            raw_diagnostics = client.get_diagnostics(rel_path, inactivity_timeout=15.0)
            check_lsp_response(raw_diagnostics, "get_diagnostics")
        finally:
            if opened_file:
                try:
                    client.close_files([rel_path])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to close `%s` after run_code: %s", rel_path, exc)
            try:
                os.remove(abs_path)
            except FileNotFoundError:
                pass
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to remove temporary Lean snippet `%s`: %s", abs_path, e)

        diagnostics = _to_diagnostic_messages(raw_diagnostics)
        has_errors = any(d.severity == "error" for d in diagnostics)
        return RunResult(success=not has_errors, diagnostics=diagnostics)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
