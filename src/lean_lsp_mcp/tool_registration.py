from __future__ import annotations

from collections.abc import Awaitable, Callable

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import CallToolResult

from lean_lsp_mcp.app_surface import AppSurfaceConfig
from lean_lsp_mcp.auth import AuthConfig
from lean_lsp_mcp.profiles import ServerProfile, write_tools_enabled
from lean_lsp_mcp.tools_app import register_app_tools
from lean_lsp_mcp.tools_external import register_external_tools
from lean_lsp_mcp.tools_read import register_read_tools
from lean_lsp_mcp.tools_write import register_write_tools


READ_TOOL_NAMES: tuple[str, ...] = (
    "app_home",
    "search",
    "fetch",
    "outline",
    "diagnostics",
    "goal",
    "term_goal",
    "hover",
    "completions",
    "declaration",
    "local_search",
    "leansearch",
    "loogle",
    "leanfinder",
    "state_search",
    "hammer_premise",
    "profile_proof",
)

WRITE_TOOL_NAMES: tuple[str, ...] = (
    "build",
    "multi_attempt",
    "run_code",
)


MixedAuthChecker = Callable[[Context, str], Awaitable[CallToolResult | None]]


def enabled_tool_names(profile: ServerProfile) -> list[str]:
    names = list(READ_TOOL_NAMES)
    if write_tools_enabled(profile):
        names.extend(WRITE_TOOL_NAMES)
    return names


def register_tools(
    mcp: FastMCP,
    *,
    app_config: AppSurfaceConfig,
    auth_config: AuthConfig,
    profile: ServerProfile,
    rg_available: bool,
    rg_message: str,
    logger,
    mixed_auth_checker: MixedAuthChecker,
) -> None:
    register_app_tools(
        mcp,
        app_config=app_config,
        profile=profile,
        auth_config=auth_config,
        read_tool_names=READ_TOOL_NAMES,
        write_tool_names=WRITE_TOOL_NAMES,
    )
    register_read_tools(mcp, rg_available=rg_available, rg_message=rg_message)
    register_external_tools(
        mcp,
        rg_available=rg_available,
        rg_message=rg_message,
        logger=logger,
    )

    if write_tools_enabled(profile):
        register_write_tools(
            mcp,
            logger=logger,
            mixed_auth_checker=mixed_auth_checker,
        )
