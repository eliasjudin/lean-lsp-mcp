from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mcp.server.fastmcp import Context
from mcp.server.fastmcp.server import FastMCP as _FastMCPType
from mcp.types import CallToolResult, TextContent
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from lean_lsp_mcp.auth import (
    AuthConfig,
    AuthMode,
    CompositeTokenVerifier,
    bearer_token_from_header,
    oauth_resource_metadata_url,
)
from lean_lsp_mcp.models import (
    BuildResult,
    DiagnosticMessage,
    MultiAttemptResult,
    RunResult,
)


def security_schemes_for_tool(
    *, auth_config: AuthConfig, write_tool_names: Iterable[str], tool_name: str
) -> list[dict[str, Any]] | None:
    oauth_scheme: dict[str, Any] = {"type": "oauth2"}
    if auth_config.required_scopes:
        oauth_scheme["scopes"] = auth_config.required_scopes

    write_tools = set(write_tool_names)

    if auth_config.mode == AuthMode.MIXED:
        if tool_name in write_tools:
            return [oauth_scheme]
        return [{"type": "noauth"}]

    if auth_config.mode in {AuthMode.OAUTH, AuthMode.OAUTH_AND_BEARER}:
        return [oauth_scheme]

    if auth_config.mode == AuthMode.NONE:
        return [{"type": "noauth"}]

    # AuthMode.BEARER does not have a standard MCP scheme descriptor.
    return None


def oauth_protected_resource_payload(auth_config: AuthConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "resource": auth_config.resource_server_url,
        "authorization_servers": [auth_config.issuer_url],
    }
    if auth_config.required_scopes:
        payload["scopes_supported"] = auth_config.required_scopes
    return payload


def register_oauth_metadata_route(
    mcp: _FastMCPType, *, auth_config: AuthConfig
) -> None:
    if auth_config.mode not in {
        AuthMode.OAUTH,
        AuthMode.OAUTH_AND_BEARER,
        AuthMode.MIXED,
    }:
        return

    @mcp.custom_route(
        "/.well-known/oauth-protected-resource",
        methods=["GET"],
        include_in_schema=False,
    )
    async def oauth_protected_resource_metadata(_request: Request) -> Response:
        return JSONResponse(oauth_protected_resource_payload(auth_config))


def authorization_header_from_context(ctx: Context) -> str | None:
    request = getattr(ctx.request_context, "request", None)
    if request is None:
        return None
    headers = getattr(request, "headers", None)
    if headers is None:
        return None
    return headers.get("authorization")


def www_authenticate_challenge(auth_config: AuthConfig) -> list[str]:
    challenge_parts: list[str] = []

    metadata_url = oauth_resource_metadata_url(auth_config)
    if metadata_url:
        challenge_parts.append(f'resource_metadata="{metadata_url}"')

    if auth_config.required_scopes:
        challenge_parts.append(f'scope="{" ".join(auth_config.required_scopes)}"')

    challenge_parts.append('error="invalid_token"')
    challenge_parts.append(
        'error_description="Authentication required for this tool. Link your account and retry."'
    )
    return [f"Bearer {', '.join(challenge_parts)}"]


def auth_required_tool_result(
    *, tool_name: str, auth_config: AuthConfig
) -> CallToolResult:
    message = "Authentication required for this tool."
    if tool_name == "build":
        structured = BuildResult(success=False, output="", errors=[message]).model_dump(
            mode="json"
        )
    elif tool_name == "multi_attempt":
        structured = MultiAttemptResult(items=[]).model_dump(mode="json")
    elif tool_name == "run_code":
        structured = RunResult(
            success=False,
            diagnostics=[
                DiagnosticMessage(
                    severity="error",
                    message=message,
                    line=1,
                    column=1,
                )
            ],
        ).model_dump(mode="json")
    else:
        structured = {"error": message}

    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        structuredContent=structured,
        isError=True,
        _meta={"mcp/www_authenticate": www_authenticate_challenge(auth_config)},
    )


async def mixed_auth_error_for_write_tool(
    ctx: Context,
    *,
    tool_name: str,
    auth_config: AuthConfig,
    token_verifier: CompositeTokenVerifier | None,
    logger,
) -> CallToolResult | None:
    if auth_config.mode != AuthMode.MIXED:
        return None
    if token_verifier is None:
        logger.warning(
            "Mixed auth mode is enabled but no token verifier is configured."
        )
        return auth_required_tool_result(tool_name=tool_name, auth_config=auth_config)

    token = bearer_token_from_header(authorization_header_from_context(ctx))
    access = await token_verifier.verify_token(token)
    if access is None:
        return auth_required_tool_result(tool_name=tool_name, auth_config=auth_config)
    return None
