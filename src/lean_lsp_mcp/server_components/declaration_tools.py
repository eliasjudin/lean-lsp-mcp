from __future__ import annotations

import os
import time
from typing import Any

from mcp.server.fastmcp import Context

from lean_lsp_mcp.file_utils import get_file_contents
from lean_lsp_mcp.schema_types import (
    ERROR_BAD_REQUEST,
    ERROR_INVALID_PATH,
    ERROR_UNKNOWN,
)
from lean_lsp_mcp.tool_inputs import LeanDeclarationFileInput
from lean_lsp_mcp.tool_spec import TOOL_ANNOTATIONS
from lean_lsp_mcp.utils import file_identity, find_start_position, uri_to_absolute_path

from .common import (
    ToolError,
    _compact_pos,
    _resource_item,
    _sanitize_path_label,
    _set_response_format_hint,
    _text_item,
    error_result,
    open_file_session,
    success_result,
)
from .context import mcp

__all__ = ["declaration_file"]


@mcp.tool(
    "lean_declaration_file",
    description="Open the Lean source file that defines a given declaration and surface declaration path metadata for navigation.",
    annotations=TOOL_ANNOTATIONS["lean_declaration_file"],
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
            if abs_path is None or not os.path.exists(abs_path):
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
