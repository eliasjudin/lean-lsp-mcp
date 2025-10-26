from __future__ import annotations

import time
from typing import Any, Dict, List

from mcp.server.fastmcp import Context

from lean_lsp_mcp.schema_types import ERROR_BAD_REQUEST, ERROR_NOT_GOAL_POSITION
from lean_lsp_mcp.tool_inputs import LeanGoalInput, LeanTermGoalInput
from lean_lsp_mcp.tool_spec import TOOL_ANNOTATIONS
from lean_lsp_mcp.utils import format_goal, goal_to_payload

from .common import (
    ToolError,
    _compact_pos,
    _set_response_format_hint,
    _text_item,
    error_result,
    open_file_session,
    success_result,
)
from .context import mcp

__all__ = ["goal", "term_goal"]


@mcp.tool(
    "lean_goal",
    description="Inspect Lean proof goals at a 1-indexed file position; omitting the column probes line boundaries and returns compact `goals` entries with first-line previews.",
    annotations=TOOL_ANNOTATIONS["lean_goal"],
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

                results: List[Dict[str, Any]] = []
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
                for result in results:
                    kind = result.get("kind")
                    pos = result.get("position", {})
                    gl = result.get("goal") or {}
                    rendered = (
                        gl.get("rendered") if isinstance(gl, dict) else None
                    )
                    raw_line = pos.get("line")
                    raw_column = pos.get("character")
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
                        column_index = (
                            column_start + 1 if kind == "line_start" else column_end + 1
                        )
                    elif column_value >= 0:
                        column_index = column_value + 1
                    else:
                        column_index = (
                            column_start + 1 if kind == "line_start" else column_end + 1
                        )

                    line_index = max(line_index, 1)
                    column_index = max(column_index, 1)
                    item: Dict[str, Any] = {
                        "k": ("start" if kind == "line_start" else "end"),
                        "p": [line_index, column_index],
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
                    details={
                        "file": identity["relative_path"],
                        "line": line,
                        "column": column,
                    },
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
    annotations=TOOL_ANNOTATIONS["lean_term_goal"],
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
