from __future__ import annotations

import asyncio
import os
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Awaitable, Callable

from leanclient import DocumentContentChange, LeanLSPClient
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import CallToolResult, ToolAnnotations
from pydantic import Field

from lean_lsp_mcp.file_utils import get_file_contents
from lean_lsp_mcp.models import (
    AttemptResult,
    BuildResult,
    DiagnosticMessage,
    MultiAttemptResult,
    RunResult,
)
from lean_lsp_mcp.repl import SnippetResult
from lean_lsp_mcp.tool_utils import resolve_and_prepare_file, to_diagnostic_messages
from lean_lsp_mcp.utils import (
    LeanToolError,
    OutputCapture,
    check_lsp_response,
    filter_diagnostics_by_position,
)


MixedAuthChecker = Callable[[Context, str], Awaitable[CallToolResult | None]]


def should_fetch_build_cache(env: Mapping[str, str] | None = None) -> bool:
    values = env if env is not None else os.environ
    raw = values.get("LEAN_BUILD_FETCH_CACHE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


async def maybe_run_cache_prefetch(
    *,
    lean_project_path_obj: Path,
    report_progress: Callable[..., Awaitable[None]],
    logger,
    create_process_exec: Callable[..., Awaitable[asyncio.subprocess.Process]],
    env: Mapping[str, str] | None = None,
) -> None:
    if not should_fetch_build_cache(env):
        return

    await report_progress(progress=2, total=16, message="Running `lake exe cache get`")
    try:
        cache_proc = await create_process_exec(
            "lake", "exe", "cache", "get", cwd=lean_project_path_obj
        )
        return_code = await cache_proc.wait()
        if return_code != 0:
            logger.warning(
                "`lake exe cache get` failed with return code %s", return_code
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("`lake exe cache get` failed: %s", exc)


async def _multi_attempt_repl(
    ctx: Context,
    abs_path: Path,
    rel_path: str,
    line: int,
    snippets: list[str],
    logger,
) -> MultiAttemptResult | None:
    app_ctx = ctx.request_context.lifespan_context
    if not app_ctx.repl_enabled or not app_ctx.repl:
        return None

    try:
        content = get_file_contents(str(abs_path))
        lines = content.splitlines()
        if line > len(lines):
            return None

        base_code = "\n".join(lines[: line - 1])
        repl_results: list[SnippetResult] = await app_ctx.repl.run_snippets(
            base_code, snippets
        )

        results: list[AttemptResult] = []
        for snippet, partial_result in zip(snippets, repl_results):
            diagnostics = [
                DiagnosticMessage(
                    severity=m.get("severity", "info"),
                    message=m.get("data", ""),
                    line=m.get("pos", {}).get("line", 0),
                    column=m.get("pos", {}).get("column", 0),
                )
                for m in (partial_result.messages or [])
            ]
            if partial_result.error:
                diagnostics.append(
                    DiagnosticMessage(
                        severity="error",
                        message=partial_result.error,
                        line=0,
                        column=0,
                    )
                )
            results.append(
                AttemptResult(
                    snippet=snippet.rstrip("\n"),
                    goals=partial_result.goals or [],
                    diagnostics=diagnostics,
                )
            )
        return MultiAttemptResult(items=results)
    except Exception as exc:  # noqa: BLE001
        logger.debug("REPL multi_attempt failed: %s", exc)
        return None


def _multi_attempt_lsp(
    ctx: Context,
    abs_path: Path,
    rel_path: str,
    line: int,
    snippets: list[str],
    logger,
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
            diagnostics = client.get_diagnostics(rel_path)
            check_lsp_response(diagnostics, "get_diagnostics")
            filtered_diag = filter_diagnostics_by_position(diagnostics, line - 1, None)
            goal_result = client.get_goal(rel_path, line - 1, len(snippet_str))
            goals = goal_result.get("goals", []) if goal_result else []
            results.append(
                AttemptResult(
                    snippet=snippet_str,
                    goals=goals,
                    diagnostics=to_diagnostic_messages(filtered_diag),
                )
            )

        return MultiAttemptResult(items=results)
    finally:
        try:
            client.update_file_content(rel_path, original_content)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to restore `%s` after multi_attempt: %s", rel_path, exc
            )


def register_write_tools(
    mcp: FastMCP,
    *,
    logger,
    mixed_auth_checker: MixedAuthChecker,
) -> None:
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
        clean: Annotated[
            bool, Field(description="Run lake clean first (slow)")
        ] = False,
        output_lines: Annotated[
            int, Field(description="Return last N lines of build log (0=none)")
        ] = 20,
    ) -> BuildResult:
        """Use this when you need to run `lake build` and inspect recent build output."""
        auth_error = await mixed_auth_checker(ctx, "build")
        if auth_error is not None:
            return auth_error

        app_ctx = ctx.request_context.lifespan_context
        lean_project_path_obj = app_ctx.lean_project_path

        log_lines: list[str] = []
        errors: list[str] = []

        try:
            client: LeanLSPClient | None = app_ctx.client
            if client:
                app_ctx.client = None
                client.close()

            if clean:
                await ctx.report_progress(
                    progress=1, total=16, message="Running `lake clean`"
                )
                clean_proc = await asyncio.create_subprocess_exec(
                    "lake", "clean", cwd=lean_project_path_obj
                )
                await clean_proc.wait()

            await maybe_run_cache_prefetch(
                lean_project_path_obj=lean_project_path_obj,
                report_progress=ctx.report_progress,
                logger=logger,
                create_process_exec=asyncio.create_subprocess_exec,
            )

            process = await asyncio.create_subprocess_exec(
                "lake",
                "build",
                "--verbose",
                cwd=lean_project_path_obj,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            assert process.stdout is not None
            while line := await process.stdout.readline():
                line_str = line.decode("utf-8", errors="replace").rstrip()

                if line_str.startswith("trace:") or "LEAN_PATH=" in line_str:
                    continue

                log_lines.append(line_str)
                if "error" in line_str.lower():
                    errors.append(line_str)

                if match := re.search(
                    r"\[(\d+)/(\d+)\]\s*(.+?)(?:\s+\(\d+\.?\d*[ms]+\))?$", line_str
                ):
                    await ctx.report_progress(
                        progress=int(match.group(1)),
                        total=int(match.group(2)),
                        message=match.group(3) or "Building",
                    )

            await process.wait()

            if process.returncode != 0:
                return BuildResult(
                    success=False,
                    output="\n".join(log_lines[-output_lines:]) if output_lines else "",
                    errors=errors
                    or [f"Build failed with return code {process.returncode}"],
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

        except Exception as exc:  # noqa: BLE001
            return BuildResult(
                success=False,
                output="\n".join(log_lines[-output_lines:]) if output_lines else "",
                errors=[str(exc)],
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
        snippets: Annotated[
            list[str], Field(description="Tactics to try (3+ recommended)")
        ],
    ) -> MultiAttemptResult:
        """Use this when you want to compare multiple tactic snippets at one line."""
        auth_error = await mixed_auth_checker(ctx, "multi_attempt")
        if auth_error is not None:
            return auth_error

        abs_path, rel_path = resolve_and_prepare_file(ctx, path)

        result = await _multi_attempt_repl(
            ctx, abs_path, rel_path, line, snippets, logger
        )
        if result is not None:
            return result

        return _multi_attempt_lsp(ctx, abs_path, rel_path, line, snippets, logger)

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
    async def run_code(
        ctx: Context,
        code: Annotated[
            str, Field(description="Self-contained Lean code with imports")
        ],
    ) -> RunResult:
        """Use this when you need diagnostics for a temporary standalone Lean snippet."""
        auth_error = await mixed_auth_checker(ctx, "run_code")
        if auth_error is not None:
            return auth_error

        app_ctx = ctx.request_context.lifespan_context
        lean_project_path = app_ctx.lean_project_path

        rel_path = f"_mcp_snippet_{uuid.uuid4().hex}.lean"
        abs_path = lean_project_path / rel_path

        try:
            with open(abs_path, "w", encoding="utf-8") as file_handle:
                file_handle.write(code)
        except Exception as exc:  # noqa: BLE001
            raise LeanToolError(f"Error writing code snippet: {exc}") from exc

        client: LeanLSPClient | None = app_ctx.client
        raw_diagnostics: list[dict] = []
        opened_file = False

        try:
            if client is None:
                from lean_lsp_mcp.client_utils import startup_client

                startup_client(ctx)
                client = app_ctx.client
                if client is None:
                    raise LeanToolError(
                        "Failed to initialize Lean client for run_code."
                    )

            client.open_file(rel_path)
            opened_file = True
            raw_diagnostics = client.get_diagnostics(rel_path, inactivity_timeout=15.0)
            check_lsp_response(raw_diagnostics, "get_diagnostics")
        finally:
            if opened_file:
                try:
                    client.close_files([rel_path])
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to close `%s` after run_code: %s", rel_path, exc
                    )
            try:
                os.remove(abs_path)
            except FileNotFoundError:
                pass
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to remove temporary Lean snippet `%s`: %s", abs_path, exc
                )

        diagnostics = to_diagnostic_messages(raw_diagnostics)
        has_errors = any(d.severity == "error" for d in diagnostics)
        return RunResult(success=not has_errors, diagnostics=diagnostics)
