from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import Context

from lean_lsp_mcp.file_utils import get_file_contents, get_relative_file_path
from lean_lsp_mcp.schema_types import ERROR_BAD_REQUEST, ERROR_INVALID_PATH
from lean_lsp_mcp.tool_inputs import (
    LeanDiagnosticMessagesInput,
    LeanFileContentsInput,
)
from lean_lsp_mcp.tool_spec import TOOL_ANNOTATIONS
from lean_lsp_mcp.utils import (
    compute_pagination,
    diagnostics_to_entries,
    file_identity,
    summarize_diagnostics,
)

from .common import (
    ToolError,
    _compact_diagnostics,
    _resource_item,
    _sanitize_path_label,
    _set_response_format_hint,
    _text_item,
    error_result,
    open_file_session,
    success_result,
)
from .context import mcp

__all__ = ["lean_file_contents", "lean_diagnostic_messages"]


def _validate_pagination_params(
    start_line: Optional[int],
    line_count: Optional[int],
    start_time: float,
    ctx: Context,
    response_format: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Validate start_line and line_count parameters.
    
    Returns error result if validation fails, None if validation passes.
    """
    if start_line is not None and start_line < 1:
        return error_result(
            message="`start_line` must be >= 1",
            code=ERROR_BAD_REQUEST,
            details={"start_line": start_line},
            start_time=start_time,
            ctx=ctx,
            response_format=response_format,
        )
    if line_count is not None and line_count < 1:
        return error_result(
            message="`line_count` must be >= 1",
            code=ERROR_BAD_REQUEST,
            details={"line_count": line_count},
            start_time=start_time,
            ctx=ctx,
            response_format=response_format,
        )
    return None


@mcp.tool(
    "lean_file_contents",
    description="Read Lean source text with optional 1-indexed annotations; `structured.lines` preserves numbering for downstream agents.",
    annotations=TOOL_ANNOTATIONS["lean_file_contents"],
)
def lean_file_contents(ctx: Context, params: LeanFileContentsInput) -> Any:
    started = time.perf_counter()
    file_path = params.file_path
    annotate_lines = params.annotate_lines
    start_line = params.start_line
    line_count = params.line_count
    response_format = params.response_format
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

    if expanded and expanded not in candidates:
        candidates.append(expanded)
    elif not expanded:
        candidates.append(expanded)

    resolved_path = next((path for path in candidates if os.path.exists(path)), None)

    if resolved_path is None:
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
        if is_relative:
            details["project_root"] = project_label
            details["candidates"] = [_sanitize_path_label(p) for p in candidates]

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

    # Validate pagination parameters
    validation_error = _validate_pagination_params(
        start_line, line_count, started, ctx, response_format
    )
    if validation_error is not None:
        return validation_error

    rel_path = None
    if project_root:
        try:
            rel_path = get_relative_file_path(project_root, resolved_path)
        except (ValueError, OSError):
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
        "file": {
            "uri": identity["uri"],
            "path": identity["relative_path"],
        },
        "slice": {
            "start": start,
            "end": end if total_lines else 0,
            "total": total_lines,
        },
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

    summary = f"{start}-{end}/{total_lines}" if total_lines else "Empty"

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
    annotations=TOOL_ANNOTATIONS["lean_diagnostic_messages"],
)
def lean_diagnostic_messages(ctx: Context, params: LeanDiagnosticMessagesInput) -> Any:
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
            # Validate pagination parameters
            validation_error = _validate_pagination_params(
                start_line, line_count, started, ctx, response_format
            )
            if validation_error is not None:
                return validation_error

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

            error_count = summary["bySeverity"].get("error", 0)
            summary_text = f"{summary['count']} diagnostics ({error_count} errors)."
            messages, diags_compact = _compact_diagnostics(entries)

            by_sev_codes: Dict[str, int] = {}
            for label, count in summary.get("bySeverity", {}).items():
                code = {"error": 1, "warning": 2, "info": 3, "hint": 4}.get(label)
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
