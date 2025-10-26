from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from threading import Lock
from typing import Any, List

from mcp.server.fastmcp import Context
from mcp.server.fastmcp.utilities.logging import get_logger

from lean_lsp_mcp.schema_types import ERROR_BAD_REQUEST, ERROR_IO_FAILURE
from lean_lsp_mcp.tool_inputs import LeanBuildInput
from lean_lsp_mcp.tool_spec import TOOL_ANNOTATIONS

from ..leanclient_provider import ensure_leanclient_available, is_leanclient_available
from ..utils import OutputCapture, log_event
from .common import (
    _resource_item,
    _sanitize_path_label,
    _set_response_format_hint,
    _text_item,
    error_result,
    sanitize_exception,
    success_result,
)
from .context import mcp

logger = get_logger(__name__)


__all__ = ["lean_build"]


@mcp.tool(
    "lean_build",
    description="Run `lake build` (optionally `lake clean`) to refresh the Lean project and restart the cached Lean LSP client.",
    annotations=TOOL_ANNOTATIONS["lean_build"],
)
def lean_build(ctx: Context, params: LeanBuildInput) -> Any:
    started = time.perf_counter()
    lean_project_path = params.lean_project_path or None
    clean = params.clean
    response_format = params.response_format
    _set_response_format_hint(ctx, response_format)

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
        return error_result(
            message=message,
            code=ERROR_BAD_REQUEST,
            start_time=started,
            ctx=ctx,
            response_format=response_format,
        )

    build_output = ""
    build_output_parts: List[str] = []
    sanitized_project = _sanitize_path_label(lean_project_path)
    project_uri = Path(lean_project_path).resolve().as_uri()

    def _record_output(text: str | None) -> None:
        if text:
            build_output_parts.append(text)

    lifespan = ctx.request_context.lifespan_context
    try:
        client = getattr(lifespan, "client", None)
        if client:
            client.close()
            file_cache_lock = getattr(lifespan, "file_cache_lock", None)
            if file_cache_lock is None:
                file_cache_lock = Lock()
                lifespan.file_cache_lock = file_cache_lock
            with file_cache_lock:
                lifespan.file_content_hashes.clear()

        if clean:
            clean_proc = subprocess.run(
                ["lake", "clean"],
                cwd=lean_project_path,
                check=False,
                capture_output=True,
                text=True,
            )
            _record_output(clean_proc.stdout)
            _record_output(clean_proc.stderr)
            log_event(
                logger,
                level=20,
                message="Ran `lake clean`",
                ctx=ctx,
                project_path=sanitized_project,
                clean=clean,
            )

        if is_leanclient_available():
            LeanClientCls, _ = ensure_leanclient_available()
            with OutputCapture() as output:
                client = LeanClientCls(
                    lean_project_path,
                    initial_build=True,
                    print_warnings=False,
                )
            lifespan.client = client
            _record_output(output.get_output())
            log_event(
                logger,
                level=20,
                message="Built project and re-started LSP client",
                ctx=ctx,
                project_path=sanitized_project,
                clean=clean,
                lsp_restarted=True,
            )
            build_output = "".join(build_output_parts)
            summary = f"Build ok (clean={str(clean).lower()})"
            structured = {
                "project": {"path": sanitized_project},
                "clean": clean,
                "status": "ok",
                "lsp_restarted": True,
            }
            content_items = [_text_item(summary)]
            if build_output and len(build_output) <= 4000:
                content_items.append(_resource_item(project_uri, build_output))
            return success_result(
                summary=summary,
                structured=structured,
                start_time=started,
                ctx=ctx,
                content=content_items,
                response_format=response_format,
            )

        build_proc = subprocess.run(
            ["lake", "build"],
            cwd=lean_project_path,
            check=False,
            capture_output=True,
            text=True,
        )
        _record_output(build_proc.stdout)
        _record_output(build_proc.stderr)
        build_output = "".join(build_output_parts)
        if build_proc.returncode != 0:
            message = "`lake build` failed"
            details = {
                "project_path": sanitized_project,
                "clean": clean,
                "exit_code": build_proc.returncode,
                "output": build_output,
            }
            return error_result(
                message=message,
                code=ERROR_IO_FAILURE,
                details=details,
                start_time=started,
                ctx=ctx,
                response_format=response_format,
            )

        lifespan.client = None
        summary = f"Build ok (clean={str(clean).lower()}, leanclient missing)"
        structured = {
            "project": {"path": sanitized_project},
            "clean": clean,
            "status": "ok",
            "lsp_restarted": False,
        }
        content_items = [_text_item(summary)]
        if build_output and len(build_output) <= 4000:
            content_items.append(_resource_item(project_uri, build_output))
        return success_result(
            summary=summary,
            structured=structured,
            start_time=started,
            ctx=ctx,
            content=content_items,
            response_format=response_format,
        )
    except Exception as exc:
        if build_output_parts:
            build_output = "".join(build_output_parts)
        sanitized_error = sanitize_exception(exc, fallback_reason="running lake build")
        message = "Build failed"
        details = {
            "project_path": sanitized_project,
            "clean": clean,
            "output": build_output,
            "error": sanitized_error,
        }
        log_event(
            logger,
            level=40,
            message="Build failed",
            ctx=ctx,
            project_path=sanitized_project,
            clean=clean,
            error=sanitized_error,
            output=build_output,
        )
        return error_result(
            message=message,
            code=ERROR_IO_FAILURE,
            details=details,
            start_time=started,
            ctx=ctx,
            response_format=response_format,
        )
