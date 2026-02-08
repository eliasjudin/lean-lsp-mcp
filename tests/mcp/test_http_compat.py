from __future__ import annotations

import asyncio
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient


def _http_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], str]:
    request = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            status = response.status
            response_headers = {k.lower(): v for k, v in response.headers.items()}
            body = response.read().decode("utf-8", errors="replace")
            return status, response_headers, body
    except urllib.error.HTTPError as exc:
        status = exc.code
        response_headers = {k.lower(): v for k, v in exc.headers.items()}
        body = exc.read().decode("utf-8", errors="replace")
        return status, response_headers, body


@pytest.mark.asyncio
async def test_health_endpoint_available(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    async with remote_client_factory(
        transport="streamable-http", profile="read", auth_mode="none"
    ) as client:
        status, _, body = await asyncio.to_thread(
            _http_request,
            f"{client.base_url}/",
        )
    assert status == 200
    assert "Lean LSP MCP server" in body


@pytest.mark.asyncio
async def test_cors_preflight_supported_for_mcp_and_nested_paths(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    async with remote_client_factory(
        transport="streamable-http", profile="read", auth_mode="none"
    ) as client:
        headers = {
            "Origin": "https://chatgpt.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,mcp-session-id",
        }

        status_mcp, response_headers_mcp, _ = await asyncio.to_thread(
            _http_request,
            f"{client.base_url}/mcp",
            method="OPTIONS",
            headers=headers,
        )
        assert status_mcp in {200, 204}
        assert response_headers_mcp.get("access-control-allow-origin") in {
            "*",
            "https://chatgpt.com",
        }

        status_nested, response_headers_nested, _ = await asyncio.to_thread(
            _http_request,
            f"{client.base_url}/mcp/actions",
            method="OPTIONS",
            headers=headers,
        )
        assert status_nested in {200, 204}
        assert response_headers_nested.get("access-control-allow-origin") in {
            "*",
            "https://chatgpt.com",
        }


@pytest.mark.asyncio
async def test_none_auth_mode_keeps_oauth_metadata_route_absent(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    async with remote_client_factory(
        transport="streamable-http", profile="read", auth_mode="none"
    ) as client:
        status, _, _ = await asyncio.to_thread(
            _http_request,
            f"{client.base_url}/.well-known/oauth-protected-resource",
        )
    assert status == 404


@pytest.mark.asyncio
async def test_transport_security_rejects_untrusted_origin_by_default(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    async with remote_client_factory(
        transport="streamable-http", profile="read", auth_mode="none"
    ) as client:
        status, _, body = await asyncio.to_thread(
            _http_request,
            f"{client.base_url}/mcp",
            headers={
                "Origin": "https://evil.example",
                "Accept": "application/json",
            },
        )
    assert status == 403
    assert "Invalid Origin header" in body
