from __future__ import annotations

import asyncio
import json
import urllib.request
from collections.abc import Callable
from typing import AsyncContextManager

import pytest

from mcp.server.auth.provider import AccessToken

from lean_lsp_mcp.auth import AuthConfig, AuthMode, CompositeTokenVerifier
from tests.helpers.mcp_client import MCPClient


@pytest.mark.asyncio
async def test_bearer_verifier_accepts_expected_token() -> None:
    verifier = CompositeTokenVerifier(
        AuthConfig(
            mode=AuthMode.BEARER,
            issuer_url=None,
            resource_server_url=None,
            required_scopes=[],
            bearer_token="secret",
        )
    )
    granted = await verifier.verify_token("secret")
    rejected = await verifier.verify_token("wrong")
    assert granted is not None
    assert granted.token == "secret"
    assert rejected is None


@pytest.mark.asyncio
async def test_oauth_and_bearer_fallback_works() -> None:
    verifier = CompositeTokenVerifier(
        AuthConfig(
            mode=AuthMode.OAUTH_AND_BEARER,
            issuer_url="https://issuer.example.invalid",
            resource_server_url="https://resource.example.invalid",
            required_scopes=[],
            bearer_token="secret",
        )
    )

    class _FakeOIDC:
        async def verify(self, token: str):
            if token == "oauth-token":
                return AccessToken(token=token, client_id="cid", scopes=[])
            return None

    verifier._oidc = _FakeOIDC()  # type: ignore[attr-defined]

    oauth_access = await verifier.verify_token("oauth-token")
    bearer_access = await verifier.verify_token("secret")
    missing = await verifier.verify_token(None)

    assert oauth_access is not None
    assert bearer_access is not None
    assert missing is None


@pytest.mark.asyncio
async def test_mixed_bearer_fallback_works() -> None:
    verifier = CompositeTokenVerifier(
        AuthConfig(
            mode=AuthMode.MIXED,
            issuer_url="https://issuer.example.invalid",
            resource_server_url="https://resource.example.invalid",
            required_scopes=[],
            bearer_token="secret",
        )
    )

    class _FakeOIDC:
        async def verify(self, token: str):
            if token == "oauth-token":
                return AccessToken(token=token, client_id="cid", scopes=[])
            return None

    verifier._oidc = _FakeOIDC()  # type: ignore[attr-defined]

    oauth_access = await verifier.verify_token("oauth-token")
    bearer_access = await verifier.verify_token("secret")
    missing = await verifier.verify_token(None)

    assert oauth_access is not None
    assert bearer_access is not None
    assert missing is None


@pytest.mark.asyncio
async def test_remote_bearer_auth_rejects_invalid_token(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    with pytest.raises(Exception):
        async with remote_client_factory(
            transport="streamable-http",
            profile="write",
            auth_mode="bearer",
            token="server-token",
            client_token="wrong-token",
        ) as client:
            await client.list_tools()


@pytest.mark.asyncio
async def test_remote_bearer_auth_accepts_valid_token(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    async with remote_client_factory(
        transport="streamable-http",
        profile="write",
        auth_mode="bearer",
        token="server-token",
        client_token="server-token",
    ) as client:
        tools = await client.list_tools()
        assert "outline" in tools


@pytest.mark.asyncio
async def test_remote_mixed_list_tools_allows_no_auth(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    async with remote_client_factory(
        transport="streamable-http",
        profile="write",
        auth_mode="mixed",
        token="server-token",
        client_token="",
    ) as client:
        tools = await client.list_tools()
        assert "search" in tools
        assert "build" in tools


@pytest.mark.asyncio
async def test_remote_mixed_list_tools_exposes_security_schemes(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    async with remote_client_factory(
        transport="streamable-http",
        profile="write",
        auth_mode="mixed",
        token="server-token",
        client_token="",
    ) as client:
        tools = await client.list_tools_full()
        payloads = {tool.name: tool.model_dump(by_alias=True) for tool in tools}

        assert payloads["search"]["securitySchemes"] == [{"type": "noauth"}]
        build_schemes = payloads["build"]["securitySchemes"]
        assert len(build_schemes) == 1
        assert build_schemes[0]["type"] == "oauth2"


@pytest.mark.asyncio
async def test_remote_none_list_tools_exposes_noauth_security_schemes(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    async with remote_client_factory(
        transport="streamable-http",
        profile="write",
        auth_mode="none",
    ) as client:
        tools = await client.list_tools_full()
        payloads = {tool.name: tool.model_dump(by_alias=True) for tool in tools}
        assert payloads["search"]["securitySchemes"] == [{"type": "noauth"}]
        assert payloads["build"]["securitySchemes"] == [{"type": "noauth"}]


@pytest.mark.asyncio
async def test_remote_oauth_and_bearer_list_tools_exposes_oauth2_security_schemes(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    async with remote_client_factory(
        transport="streamable-http",
        profile="write",
        auth_mode="oauth_and_bearer",
        token="server-token",
        client_token="server-token",
    ) as client:
        tools = await client.list_tools_full()
        payloads = {tool.name: tool.model_dump(by_alias=True) for tool in tools}
        assert payloads["search"]["securitySchemes"][0]["type"] == "oauth2"
        assert payloads["build"]["securitySchemes"][0]["type"] == "oauth2"


@pytest.mark.asyncio
async def test_remote_mixed_unauthorized_write_returns_auth_challenge(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    async with remote_client_factory(
        transport="streamable-http",
        profile="write",
        auth_mode="mixed",
        token="server-token",
        client_token="",
    ) as client:
        result = await client.call_tool(
            "build",
            {"clean": False, "output_lines": 0},
            expect_error=True,
        )
        assert result.isError
        assert result.structuredContent is not None
        assert result.structuredContent["success"] is False

        meta = result.meta or {}
        assert "mcp/www_authenticate" in meta
        challenges = meta["mcp/www_authenticate"]
        assert isinstance(challenges, list)
        assert challenges
        assert "resource_metadata=" in challenges[0]


@pytest.mark.asyncio
async def test_remote_mixed_authorized_write_allows_execution(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    async with remote_client_factory(
        transport="streamable-http",
        profile="write",
        auth_mode="mixed",
        token="server-token",
        client_token="server-token",
    ) as client:
        result = await client.call_tool("run_code", {"code": "#eval Nat.succ 0\n"})
        assert result.structuredContent is not None
        assert "success" in result.structuredContent


@pytest.mark.asyncio
async def test_remote_mixed_oauth_metadata_route_is_available(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    async with remote_client_factory(
        transport="streamable-http",
        profile="write",
        auth_mode="mixed",
        token="server-token",
        client_token="",
    ) as client:
        assert client.base_url is not None
        url = f"{client.base_url}/.well-known/oauth-protected-resource"

        def _read_metadata() -> dict[str, object]:
            with urllib.request.urlopen(url, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))

        payload = await asyncio.to_thread(_read_metadata)
        assert payload["resource"] == client.base_url
        assert payload["authorization_servers"] == ["https://issuer.example.invalid"]
