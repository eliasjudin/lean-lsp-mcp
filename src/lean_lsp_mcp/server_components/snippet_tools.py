from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List

from mcp.server.fastmcp import Context

from lean_lsp_mcp.client_utils import startup_client
from lean_lsp_mcp.leanclient_provider import ensure_leanclient_available
from lean_lsp_mcp.schema_types import (
    ERROR_BAD_REQUEST,
    ERROR_CLIENT_NOT_READY,
    ERROR_IO_FAILURE,
    DiagnosticEntry,
)
from lean_lsp_mcp.tool_inputs import LeanMultiAttemptInput, LeanRunCodeInput
from lean_lsp_mcp.tool_spec import TOOL_ANNOTATIONS
from lean_lsp_mcp.utils import (
    diagnostics_to_entries,
    format_diagnostics,
    goal_to_payload,
    summarize_diagnostics,
)

from .common import (
    ToolError,
    _compact_diagnostics,
    _identity_for_rel_path,
    _resource_item,
    _sanitize_path_label,
    _set_response_format_hint,
    _text_item,
    client_session,
    error_result,
    open_file_session,
    sanitize_exception,
    success_result,
)
from .context import mcp

__all__ = ["multi_attempt", "run_code"]


@mcp.tool(
    "lean_multi_attempt",
    description="Evaluate multiple Lean snippets at a line and compare diagnostics/goals.",
    annotations=TOOL_ANNOTATIONS["lean_multi_attempt"],
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
                    goal_payload = att.get("goal")
                    if goal_payload is not None and not isinstance(goal_payload, dict):
                        return error_result(
                            message="Unexpected goal payload returned by Lean client.",
                            code=ERROR_IO_FAILURE,
                            category="lean_multi_attempt",
                            details={
                                "line": line,
                                "snippet": att.get("snippet", ""),
                                "goal_payload_type": type(goal_payload).__name__,
                            },
                            start_time=started,
                            ctx=ctx,
                        )
                    safe_goal_payload = goal_payload or {}
                    rendered_goal = safe_goal_payload.get("rendered")
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
                    "pos": {"l": max(line, 1)},
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
                except (OSError, RuntimeError):  # pragma: no cover - logged by Lean client
                    pass
    except ToolError as exc:
        return exc.payload


@mcp.tool(
    "lean_run_code",
    description="Execute an isolated Lean snippet with standalone diagnostics.",
    annotations=TOOL_ANNOTATIONS["lean_run_code"],
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
        with open(abs_path, "w") as snippet_file:
            snippet_file.write(code)
    except Exception as exc:
        error_message = sanitize_exception(exc, fallback_reason="writing snippet")
        return error_result(
            message="Error writing code snippet",
            code=ERROR_IO_FAILURE,
            category="lean_run_code",
            details={"path": _sanitize_path_label(abs_path), "error": error_message},
            start_time=started,
            ctx=ctx,
        )

    try:
        startup_client(ctx)
    except Exception as exc:
        start_error = sanitize_exception(exc, fallback_reason="starting Lean client")
        try:
            os.remove(abs_path)
        except (OSError, FileNotFoundError):
            pass
        return error_result(
            message="Error starting Lean client",
            code=ERROR_CLIENT_NOT_READY,
            category="lean_run_code",
            details={"path": _sanitize_path_label(abs_path), "error": start_error},
            start_time=started,
            ctx=ctx,
        )

    diagnostics_payload: List[DiagnosticEntry] | None = None
    formatted_diagnostics: List[str] | None = None
    close_error: str | None = None
    remove_error: str | None = None

    with client_session(ctx) as client:
        if client is None:
            try:
                os.remove(abs_path)
            except (OSError, FileNotFoundError):
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
            except (OSError, RuntimeError) as exc:  # pragma: no cover - log only
                close_error = sanitize_exception(exc, fallback_reason="closing Lean document")
            try:
                os.remove(abs_path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                remove_error = sanitize_exception(exc, fallback_reason="removing temporary file")

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
    messages, diags_compact = _compact_diagnostics(diagnostics_entries)

    by_sev_codes: Dict[str, int] = {}
    for sev_label, count in diag_summary.get("bySeverity", {}).items():
        severity_code = {"error": 1, "warning": 2, "info": 3, "hint": 4}.get(sev_label)
        if severity_code is not None:
            by_sev_codes[str(severity_code)] = count

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
