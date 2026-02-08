from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from lean_lsp_mcp.auth import AuthConfig
from lean_lsp_mcp.models import AppHomeResult, AppToolGroups, AppTransportPaths
from lean_lsp_mcp.profiles import ServerProfile, write_tools_enabled

DEFAULT_APP_NAME = "Lean LSP MCP"
DEFAULT_APP_HOME_RESOURCE_URI = "resource://lean-lsp-mcp/app/home"
DEFAULT_APP_HOME_TEMPLATE_URI = "resource://lean-lsp-mcp/app/home/{view}"


@dataclass(frozen=True, slots=True)
class AppSurfaceConfig:
    app_name: str = DEFAULT_APP_NAME
    resource_uri: str = DEFAULT_APP_HOME_RESOURCE_URI
    template_uri: str = DEFAULT_APP_HOME_TEMPLATE_URI
    streamable_http_path: str = "/mcp"
    sse_path: str = "/sse"


def _normalize_transport_path(path: str, fallback: str) -> str:
    normalized = path.strip() or fallback
    if not normalized.startswith("/"):
        return f"/{normalized}"
    return normalized


def app_surface_config_from_server(mcp: FastMCP) -> AppSurfaceConfig:
    return AppSurfaceConfig(
        streamable_http_path=_normalize_transport_path(
            mcp.settings.streamable_http_path, "/mcp"
        ),
        sse_path=_normalize_transport_path(mcp.settings.sse_path, "/sse"),
    )


def build_app_home_result(
    *,
    app_config: AppSurfaceConfig,
    profile: ServerProfile,
    auth_config: AuthConfig,
    workspace_root: Path | str,
    read_tool_names: Sequence[str],
    write_tool_names: Sequence[str],
) -> AppHomeResult:
    read_names = list(read_tool_names)
    write_names = list(write_tool_names)
    enabled_names = list(read_names)
    if write_tools_enabled(profile):
        enabled_names.extend(write_names)

    return AppHomeResult(
        app_name=app_config.app_name,
        profile=profile.value,
        auth_mode=auth_config.mode.value,
        workspace_root=str(Path(workspace_root)),
        template_uri=app_config.template_uri,
        transport_paths=AppTransportPaths(
            streamable_http=app_config.streamable_http_path,
            sse=app_config.sse_path,
        ),
        tool_groups=AppToolGroups(
            read=read_names,
            write=write_names,
            enabled=enabled_names,
        ),
    )


def _tool_list_markdown(names: Sequence[str]) -> str:
    if not names:
        return "_none_"
    return ", ".join(f"`{name}`" for name in names)


def _workspace_root_from_env() -> Path:
    root_raw = os.environ.get("LEAN_WORKSPACE_ROOT", "").strip()
    if root_raw:
        return Path(root_raw).expanduser().resolve()
    return Path.cwd()


def render_app_home_markdown(app_home: AppHomeResult) -> str:
    metadata_json = app_home.model_dump_json(indent=2)
    return "\n".join(
        [
            f"# {app_home.app_name}",
            "",
            "## Metadata",
            "```json",
            metadata_json,
            "```",
            "",
            "## Server",
            f"- Profile: `{app_home.profile}`",
            f"- Auth mode: `{app_home.auth_mode}`",
            f"- Workspace root: `{app_home.workspace_root}`",
            "",
            "## App Surface",
            f"- Home template URI: `{app_home.template_uri}`",
            (
                "- Transport paths: "
                f"`{app_home.transport_paths.streamable_http}` (streamable-http), "
                f"`{app_home.transport_paths.sse}` (sse)"
            ),
            "",
            "## Tools",
            f"- Read: {_tool_list_markdown(app_home.tool_groups.read)}",
            f"- Write: {_tool_list_markdown(app_home.tool_groups.write)}",
            f"- Enabled: {_tool_list_markdown(app_home.tool_groups.enabled)}",
        ]
    )


def register_app_home_resource(
    mcp: FastMCP,
    *,
    app_config: AppSurfaceConfig,
    profile: ServerProfile,
    auth_config: AuthConfig,
    read_tool_names: Sequence[str],
    write_tool_names: Sequence[str],
) -> None:
    def _render_home_markdown() -> str:
        app_home = build_app_home_result(
            app_config=app_config,
            profile=profile,
            auth_config=auth_config,
            workspace_root=_workspace_root_from_env(),
            read_tool_names=read_tool_names,
            write_tool_names=write_tool_names,
        )
        return render_app_home_markdown(app_home)

    @mcp.resource(
        app_config.resource_uri,
        name="app-home",
        title="Lean LSP MCP App Home",
        description="Markdown overview of Lean LSP MCP app metadata and tool groups.",
        mime_type="text/markdown",
    )
    def app_home_resource() -> str:
        return _render_home_markdown()

    @mcp.resource(
        app_config.template_uri,
        name="app-home-template",
        title="Lean LSP MCP App Home Template",
        description="Parameterized app-home template URI for MCP app host discovery.",
        mime_type="text/markdown",
    )
    def app_home_template(view: str) -> str:
        _ = view
        return _render_home_markdown()
