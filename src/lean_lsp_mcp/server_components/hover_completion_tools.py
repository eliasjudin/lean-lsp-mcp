from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from mcp.server.fastmcp import Context

from lean_lsp_mcp.tool_inputs import LeanCompletionsInput, LeanHoverInput
from lean_lsp_mcp.tool_spec import TOOL_ANNOTATIONS
from lean_lsp_mcp.utils import (
    diagnostics_to_entries,
    extract_range,
    filter_diagnostics_by_position,
    format_line,
    normalize_range,
)

from .common import (
    ToolError,
    _compact_diagnostics,
    _compact_pos,
    _normalize_range_to_arrays,
    _set_response_format_hint,
    _text_item,
    error_result,
    open_file_session,
    success_result,
)
from .context import mcp

__all__ = ["hover", "completions"]


@mcp.tool(
    "lean_hover_info",
    description="Retrieve Lean hover documentation for the symbol under the cursor together with `structured.infoSnippet` and nearby diagnostics.",
    annotations=TOOL_ANNOTATIONS["lean_hover_info"],
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
            rng = normalize_range(h_range)
            s_arr, e_arr = _normalize_range_to_arrays(rng)
            messages, diags_compact = _compact_diagnostics(diagnostic_entries)

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
    annotations=TOOL_ANNOTATIONS["lean_completions"],
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

            total_available = len(labels)
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
                "pagination": {
                    "count": len(suggestions),
                    "total": total_available,
                    "limit": max_completions,
                    "has_more": total_available > max_completions,
                },
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
