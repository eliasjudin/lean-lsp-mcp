from __future__ import annotations

import logging.config
import os
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import orjson
from leanclient import LeanLSPClient
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase
from mcp.server.fastmcp.utilities.logging import configure_logging, get_logger
from mcp.types import Tool as MCPTool
from pydantic import ConfigDict
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from lean_lsp_mcp.app_surface import (
    app_surface_config_from_server,
    register_app_home_resource,
)
from lean_lsp_mcp.auth import auth_settings_and_verifier
from lean_lsp_mcp.auth_routes import (
    mixed_auth_error_for_write_tool,
    register_oauth_metadata_route,
    security_schemes_for_tool,
)
from lean_lsp_mcp.http_config import (
    bind_host_from_env,
    bind_port_from_env,
    build_transport_security,
    load_cors_config,
)
from lean_lsp_mcp.loogle import LoogleManager
from lean_lsp_mcp.profiles import ServerProfile, get_server_profile
from lean_lsp_mcp.repl import Repl, repl_enabled
from lean_lsp_mcp.search_utils import check_ripgrep_status
from lean_lsp_mcp.tool_registration import (
    READ_TOOL_NAMES,
    WRITE_TOOL_NAMES,
    enabled_tool_names,
    register_tools,
)

INSTRUCTIONS = """## Lean LSP MCP
- Remote-first server. Use project-relative `path` only.
- Single workspace root per process.
- Use `search`/`fetch` for OpenAI MCP-compatible document retrieval.
- Read tools are always available; write tools depend on server profile.
"""

# Enforce strict JSON argument validation for all tools (no extra keys).
ArgModelBase.model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
ArgModelBase.model_rebuild(force=True)

_LOG_FILE_CONFIG = os.environ.get("LEAN_LOG_FILE_CONFIG", None)
_LOG_LEVEL = os.environ.get("LEAN_LOG_LEVEL", "INFO")
if _LOG_FILE_CONFIG:
    try:
        if _LOG_FILE_CONFIG.endswith((".yaml", ".yml")):
            import yaml

            with open(_LOG_FILE_CONFIG, "r", encoding="utf-8") as file_handle:
                cfg = yaml.safe_load(file_handle)
            logging.config.dictConfig(cfg)
        elif _LOG_FILE_CONFIG.endswith(".json"):
            with open(_LOG_FILE_CONFIG, "r", encoding="utf-8") as file_handle:
                cfg = orjson.loads(file_handle.read())
            logging.config.dictConfig(cfg)
        else:
            logging.config.fileConfig(_LOG_FILE_CONFIG, disable_existing_loggers=False)
    except Exception as exc:  # noqa: BLE001
        configure_logging("CRITICAL" if _LOG_LEVEL == "NONE" else _LOG_LEVEL)
        logger = get_logger(__name__)
        logger.warning(
            "Failed to load logging config %s: %s. Falling back to LEAN_LOG_LEVEL.",
            _LOG_FILE_CONFIG,
            exc,
        )
else:
    configure_logging("CRITICAL" if _LOG_LEVEL == "NONE" else _LOG_LEVEL)

logger = get_logger(__name__)

BIND_HOST = bind_host_from_env()
BIND_PORT = bind_port_from_env()
CORS_CONFIG = load_cors_config()
TRANSPORT_SECURITY = build_transport_security(BIND_HOST, logger=logger)

_RG_AVAILABLE, _RG_MESSAGE = check_ripgrep_status()
SERVER_PROFILE = get_server_profile()


@dataclass
class AppContext:
    workspace_root: Path
    lean_project_path: Path
    client: LeanLSPClient | None
    rate_limit: dict[str, deque[float]]
    lean_search_available: bool
    loogle_manager: LoogleManager | None = None
    loogle_local_available: bool = False
    repl: Repl | None = None
    repl_enabled: bool = False
    profile: ServerProfile = ServerProfile.READ


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    del server
    loogle_manager: LoogleManager | None = None
    loogle_local_available = False
    repl: Repl | None = None
    repl_on = False
    context: AppContext | None = None

    root_raw = os.environ.get("LEAN_WORKSPACE_ROOT", "").strip()
    if not root_raw:
        raise RuntimeError(
            "LEAN_WORKSPACE_ROOT is required (single-tenant workspace mode)."
        )

    workspace_root = Path(root_raw).expanduser().resolve()
    if not workspace_root.exists() or not workspace_root.is_dir():
        raise RuntimeError(f"LEAN_WORKSPACE_ROOT does not exist: {workspace_root}")
    if not (workspace_root / "lean-toolchain").exists():
        raise RuntimeError(
            "LEAN_WORKSPACE_ROOT must point to a Lean project root containing lean-toolchain."
        )

    try:
        if os.environ.get("LEAN_LOOGLE_LOCAL", "").lower() in ("1", "true", "yes"):
            logger.info("Local loogle enabled, initializing...")
            loogle_manager = LoogleManager(project_path=workspace_root)
            if loogle_manager.ensure_installed():
                if await loogle_manager.start():
                    loogle_local_available = True
                    logger.info("Local loogle started successfully")
                else:
                    logger.warning("Local loogle failed to start, will use remote API")
            else:
                logger.warning("Local loogle installation failed, will use remote API")

        if repl_enabled():
            from lean_lsp_mcp.repl import find_repl_binary

            repl_bin = find_repl_binary(str(workspace_root))
            if repl_bin:
                logger.info("REPL enabled, using: %s", repl_bin)
                repl = Repl(project_dir=str(workspace_root), repl_path=repl_bin)
                repl_on = True
                logger.info("REPL initialized: timeout=%ds", repl.timeout)
            else:
                logger.warning(
                    "REPL enabled but binary not found. "
                    'Add `require repl from git "https://github.com/leanprover-community/repl"` '
                    "to lakefile and run `lake build repl`. Falling back to LSP."
                )

        context = AppContext(
            workspace_root=workspace_root,
            lean_project_path=workspace_root,
            client=None,
            rate_limit={
                "leansearch": deque(),
                "loogle": deque(),
                "leanfinder": deque(),
                "lean_state_search": deque(),
                "hammer_premise": deque(),
            },
            lean_search_available=_RG_AVAILABLE,
            loogle_manager=loogle_manager,
            loogle_local_available=loogle_local_available,
            repl=repl,
            repl_enabled=repl_on,
            profile=SERVER_PROFILE,
        )
        logger.info(
            "lean-lsp-mcp started: profile=%s workspace=%s tools=%s",
            SERVER_PROFILE.value,
            workspace_root,
            ",".join(enabled_tool_names(SERVER_PROFILE)),
        )
        yield context
    finally:
        logger.info("Closing Lean LSP client")
        if context is not None and context.client:
            context.client.close()
        if loogle_manager:
            await loogle_manager.stop()
        if repl:
            await repl.close()


AUTH_CONFIG, auth_settings, token_verifier = auth_settings_and_verifier()


class LeanFastMCP(FastMCP):
    """FastMCP wrapper adding strict input schema and per-tool securitySchemes."""

    async def list_tools(self) -> list[MCPTool]:
        tools = await super().list_tools()
        result: list[MCPTool] = []
        for tool in tools:
            payload = tool.model_dump(by_alias=True, exclude_none=True)

            input_schema = payload.get("inputSchema")
            if isinstance(input_schema, dict) and input_schema.get("type") == "object":
                input_schema["additionalProperties"] = False

            security_schemes = security_schemes_for_tool(
                auth_config=AUTH_CONFIG,
                write_tool_names=WRITE_TOOL_NAMES,
                tool_name=tool.name,
            )
            if security_schemes is not None:
                payload["securitySchemes"] = security_schemes
            result.append(MCPTool.model_validate(payload))
        return result

    def _with_http_middleware(self, app):
        if getattr(app.state, "lean_http_middleware_applied", False):
            return app

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(CORS_CONFIG.allow_origins),
            allow_origin_regex=CORS_CONFIG.allow_origin_regex,
            allow_methods=list(CORS_CONFIG.allow_methods),
            allow_headers=list(CORS_CONFIG.allow_headers),
            expose_headers=list(CORS_CONFIG.expose_headers),
            allow_credentials=CORS_CONFIG.allow_credentials,
            max_age=CORS_CONFIG.max_age,
        )
        app.state.lean_http_middleware_applied = True
        return app

    def streamable_http_app(self):
        return self._with_http_middleware(super().streamable_http_app())

    def sse_app(self, mount_path: str | None = None):
        return self._with_http_middleware(super().sse_app(mount_path))


mcp_kwargs = dict(
    name="Lean LSP",
    instructions=INSTRUCTIONS,
    dependencies=["leanclient"],
    lifespan=app_lifespan,
    host=BIND_HOST,
    port=BIND_PORT,
    transport_security=TRANSPORT_SECURITY,
)
if auth_settings and token_verifier:
    mcp_kwargs["auth"] = auth_settings
    mcp_kwargs["token_verifier"] = token_verifier

mcp = LeanFastMCP(**mcp_kwargs)
APP_SURFACE_CONFIG = app_surface_config_from_server(mcp)

register_app_home_resource(
    mcp,
    app_config=APP_SURFACE_CONFIG,
    profile=SERVER_PROFILE,
    auth_config=AUTH_CONFIG,
    read_tool_names=READ_TOOL_NAMES,
    write_tool_names=WRITE_TOOL_NAMES,
)

register_oauth_metadata_route(mcp, auth_config=AUTH_CONFIG)


@mcp.custom_route("/", methods=["GET"], include_in_schema=False)
async def health(_request: Request) -> Response:
    return PlainTextResponse("Lean LSP MCP server")


async def _mixed_auth_checker(
    ctx: Context,
    tool_name: str,
):
    return await mixed_auth_error_for_write_tool(
        ctx,
        tool_name=tool_name,
        auth_config=AUTH_CONFIG,
        token_verifier=token_verifier,
        logger=logger,
    )


register_tools(
    mcp,
    app_config=APP_SURFACE_CONFIG,
    auth_config=AUTH_CONFIG,
    profile=SERVER_PROFILE,
    rg_available=_RG_AVAILABLE,
    rg_message=_RG_MESSAGE,
    logger=logger,
    mixed_auth_checker=_mixed_auth_checker,
)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
