from __future__ import annotations

from pathlib import Path
from typing import Annotated
from urllib.parse import unquote, urlparse

from leanclient import LeanLSPClient
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, Field

from lean_lsp_mcp.file_utils import get_file_contents
from lean_lsp_mcp.models import (
    CompletionItem,
    CompletionsResult,
    DeclarationInfo,
    DiagnosticsResult,
    FileOutline,
    GoalState,
    HoverInfo,
    LocalSearchResult,
    LocalSearchResults,
    ProofProfileResult,
    TermGoalState,
)
from lean_lsp_mcp.outline_utils import generate_outline_data
from lean_lsp_mcp.search_fetch import (
    declaration_text_for_id,
    search_payload_from_local_results,
)
from lean_lsp_mcp.search_utils import lean_local_search
from lean_lsp_mcp.tool_utils import (
    diagnostics_success_from_items,
    process_diagnostics,
    resolve_and_prepare_file,
    to_diagnostic_messages,
)
from lean_lsp_mcp.utils import (
    COMPLETION_KIND,
    LeanToolError,
    check_lsp_response,
    extract_goals_list,
    extract_range,
    filter_diagnostics_by_position,
    find_start_position,
    get_declaration_range,
)
from lean_lsp_mcp.pathing import PathResolutionError, to_workspace_relative


def declaration_uri_to_path(uri: str) -> Path:
    """Resolve an LSP declaration URI/path into a local filesystem path."""
    if not uri:
        raise LeanToolError("Declaration response is missing a target URI.")

    parsed = urlparse(uri)
    if parsed.scheme == "file":
        if parsed.netloc and parsed.netloc != "localhost":
            path = f"//{parsed.netloc}{unquote(parsed.path)}"
        else:
            path = unquote(parsed.path)
        return Path(path).resolve()

    if parsed.scheme == "":
        return Path(uri).resolve()

    raise LeanToolError(f"Unsupported declaration URI scheme: {parsed.scheme}")


def _text_json_result(payload: BaseModel) -> CallToolResult:
    payload_json = payload.model_dump_json()
    return CallToolResult(
        content=[TextContent(type="text", text=payload_json)],
        structuredContent=payload.model_dump(mode="json"),
    )


def register_read_tools(
    mcp: FastMCP,
    *,
    rg_available: bool,
    rg_message: str,
) -> None:
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
    ) -> CallToolResult:
        """Use this when you need candidate Lean declarations for a text query."""
        app_ctx = ctx.request_context.lifespan_context
        if not rg_available:
            raise LeanToolError(rg_message)

        raw = lean_local_search(
            query=query.strip(), limit=20, project_root=app_ctx.workspace_root
        )
        local = LocalSearchResults(
            items=[
                LocalSearchResult(name=r["name"], kind=r["kind"], file=r["file"])
                for r in raw
            ]
        )
        payload = search_payload_from_local_results(local, app_ctx.workspace_root)
        return _text_json_result(payload)

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
    ) -> CallToolResult:
        """Use this when you need the full declaration payload for a search result id."""
        app_ctx = ctx.request_context.lifespan_context
        client: LeanLSPClient | None = app_ctx.client
        if client is None:
            from lean_lsp_mcp.client_utils import startup_client

            startup_client(ctx)
            client = app_ctx.client
        if client is None:
            raise LeanToolError("Failed to initialize Lean client for fetch.")

        payload = declaration_text_for_id(
            workspace_root=app_ctx.workspace_root,
            client=client,
            identifier=id,
        )
        return _text_json_result(payload)

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
        """Use this when you need imports and declarations for a Lean source file."""
        _, rel_path = resolve_and_prepare_file(ctx, path)
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
        start_line: Annotated[
            int | None, Field(description="Filter from line", ge=1)
        ] = None,
        end_line: Annotated[
            int | None, Field(description="Filter to line", ge=1)
        ] = None,
        declaration_name: Annotated[
            str | None, Field(description="Filter to declaration (slow)")
        ] = None,
    ) -> DiagnosticsResult:
        """Use this when you need Lean diagnostics, optionally narrowed by lines or declaration."""
        _, rel_path = resolve_and_prepare_file(ctx, path)
        client: LeanLSPClient = ctx.request_context.lifespan_context.client
        client.open_file(rel_path)

        if declaration_name:
            decl_range = get_declaration_range(client, rel_path, declaration_name)
            if decl_range is None:
                raise LeanToolError(
                    f"Declaration '{declaration_name}' not found in file."
                )
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
            success = diagnostics_success_from_items(diagnostics_list)
            return process_diagnostics(diagnostics_list, success)

        diagnostics_list = getattr(result, "diagnostics", None)
        if not isinstance(diagnostics_list, list):
            raise LeanToolError(
                "Unexpected diagnostics response shape from Lean client."
            )
        success_attr = getattr(result, "success", None)
        success = (
            bool(success_attr)
            if isinstance(success_attr, bool)
            else diagnostics_success_from_items(diagnostics_list)
        )
        return process_diagnostics(diagnostics_list, success)

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
        """Use this when you need proof goals at a file position or line boundaries."""
        _, rel_path = resolve_and_prepare_file(ctx, path)
        client: LeanLSPClient = ctx.request_context.lifespan_context.client
        client.open_file(rel_path)
        content = client.get_file_content(rel_path)
        lines = content.splitlines()

        if line < 1 or line > len(lines):
            raise LeanToolError(
                f"Line {line} out of range (file has {len(lines)} lines)"
            )

        line_context = lines[line - 1]

        if column is None:
            column_end = len(line_context)
            column_start = next(
                (i for i, c in enumerate(line_context) if not c.isspace()), 0
            )
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
        return GoalState(
            line_context=line_context, goals=extract_goals_list(goal_result)
        )

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
        column: Annotated[
            int | None, Field(description="Column (defaults to end of line)", ge=1)
        ] = None,
    ) -> TermGoalState:
        """Use this when you need the expected type for a term at a cursor position."""
        _, rel_path = resolve_and_prepare_file(ctx, path)
        client: LeanLSPClient = ctx.request_context.lifespan_context.client
        client.open_file(rel_path)
        content = client.get_file_content(rel_path)
        lines = content.splitlines()

        if line < 1 or line > len(lines):
            raise LeanToolError(
                f"Line {line} out of range (file has {len(lines)} lines)"
            )

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
        column: Annotated[
            int, Field(description="Column at START of identifier", ge=1)
        ],
    ) -> HoverInfo:
        """Use this when you need symbol docs, type info, and local diagnostics at a position."""
        _, rel_path = resolve_and_prepare_file(ctx, path)
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
            diagnostics=to_diagnostic_messages(filtered),
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
        max_completions: Annotated[
            int, Field(description="Max completions", ge=1)
        ] = 32,
    ) -> CompletionsResult:
        """Use this when you need Lean IDE completions at a cursor position."""
        _, rel_path = resolve_and_prepare_file(ctx, path)
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
                import re

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
        symbol: Annotated[
            str, Field(description="Symbol (case sensitive, must be in file)")
        ],
    ) -> DeclarationInfo:
        """Use this when you need the source file and contents for a specific symbol."""
        _, rel_path = resolve_and_prepare_file(ctx, path)
        client: LeanLSPClient = ctx.request_context.lifespan_context.client
        client.open_file(rel_path)
        orig_file_content = client.get_file_content(rel_path)

        position = find_start_position(orig_file_content, symbol)
        if not position:
            raise LeanToolError(
                f"Symbol `{symbol}` (case sensitive) not found in file. Add it first."
            )

        declaration_result = client.get_declarations(
            rel_path, position["line"], position["column"]
        )

        if len(declaration_result) == 0:
            raise LeanToolError(f"No declaration available for `{symbol}`.")

        decl = declaration_result[0]
        uri = decl.get("targetUri") or decl.get("uri")
        abs_path = declaration_uri_to_path(uri)
        if not abs_path.exists():
            raise LeanToolError(
                f"Could not open declaration file `{abs_path}` for `{symbol}`."
            )

        try:
            workspace_rel_path = to_workspace_relative(
                ctx.request_context.lifespan_context.workspace_root, abs_path
            )
        except PathResolutionError as exc:
            raise LeanToolError(str(exc)) from exc

        file_content = get_file_contents(str(abs_path))
        return DeclarationInfo(path=workspace_rel_path, content=file_content)

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
        line: Annotated[
            int, Field(description="Line where theorem starts (1-indexed)", ge=1)
        ],
        top_n: Annotated[
            int, Field(description="Number of slowest lines to return", ge=1)
        ] = 5,
        timeout: Annotated[
            float, Field(description="Max seconds to wait", ge=1)
        ] = 60.0,
    ) -> ProofProfileResult:
        """Use this when you need timing hotspots for a theorem proof."""
        from lean_lsp_mcp.profile_utils import profile_theorem

        abs_path, _ = resolve_and_prepare_file(ctx, path)
        app_ctx = ctx.request_context.lifespan_context

        try:
            return await profile_theorem(
                file_path=abs_path,
                theorem_line=line,
                project_path=app_ctx.lean_project_path,
                timeout=timeout,
                top_n=top_n,
            )
        except (ValueError, TimeoutError) as exc:
            raise LeanToolError(str(exc)) from exc
