from __future__ import annotations

from collections.abc import Sequence

import os

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

from lean_lsp_mcp.app_surface import AppSurfaceConfig, build_app_home_result
from lean_lsp_mcp.auth import AuthConfig
from lean_lsp_mcp.models import AppHomeResult
from lean_lsp_mcp.profiles import ServerProfile


def register_app_tools(
    mcp: FastMCP,
    *,
    app_config: AppSurfaceConfig,
    profile: ServerProfile,
    auth_config: AuthConfig,
    read_tool_names: Sequence[str],
    write_tool_names: Sequence[str],
) -> None:
    transport = os.environ.get("LEAN_TRANSPORT", "streamable-http")

    @mcp.tool(
        "app_home",
        annotations=ToolAnnotations(
            title="App Home",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    def app_home(ctx: Context) -> AppHomeResult:
        """Use this when you need Lean LSP MCP app metadata and the home resource URI."""
        app_ctx = ctx.request_context.lifespan_context
        return build_app_home_result(
            app_config=app_config,
            profile=profile,
            auth_config=auth_config,
            workspace_root=app_ctx.workspace_root,
            read_tool_names=read_tool_names,
            write_tool_names=write_tool_names,
            transport=transport,
        )
