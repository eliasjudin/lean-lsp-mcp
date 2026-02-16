from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import Context

from lean_lsp_mcp.client_utils import startup_client
from lean_lsp_mcp.file_utils import get_relative_file_path
from lean_lsp_mcp.models import DiagnosticMessage, DiagnosticsResult
from lean_lsp_mcp.pathing import PathResolutionError, resolve_workspace_path
from lean_lsp_mcp.utils import (
    LeanToolError,
    extract_failed_dependency_paths,
    is_build_stderr,
)


DIAGNOSTIC_SEVERITY: dict[int, str] = {1: "error", 2: "warning", 3: "info", 4: "hint"}


def resolve_and_prepare_file(ctx: Context, path: str) -> tuple[Path, str]:
    app_ctx = ctx.request_context.lifespan_context
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


async def safe_report_progress(
    ctx: Context, *, progress: int, total: int, message: str
) -> None:
    try:
        await ctx.report_progress(progress=progress, total=total, message=message)
    except Exception:
        return


def to_diagnostic_messages(diagnostics: list[dict]) -> list[DiagnosticMessage]:
    result: list[DiagnosticMessage] = []
    for diag in diagnostics:
        r = diag.get("fullRange", diag.get("range"))
        if r is None:
            continue
        severity_int = diag.get("severity", 1)
        result.append(
            DiagnosticMessage(
                severity=DIAGNOSTIC_SEVERITY.get(
                    severity_int, f"unknown({severity_int})"
                ),
                message=diag.get("message", ""),
                line=r["start"]["line"] + 1,
                column=r["start"]["character"] + 1,
            )
        )
    return result


def diagnostics_success_from_items(diagnostics: list[dict]) -> bool:
    """Compute success flag for leanclient versions returning plain diagnostic lists."""
    for diag in diagnostics:
        if not isinstance(diag, dict):
            continue
        if diag.get("severity", 1) == 1:
            return False
    return True


def process_diagnostics(
    diagnostics: list[dict], build_success: bool
) -> DiagnosticsResult:
    items: list[DiagnosticMessage] = []
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
                severity=DIAGNOSTIC_SEVERITY.get(
                    severity_int, f"unknown({severity_int})"
                ),
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
