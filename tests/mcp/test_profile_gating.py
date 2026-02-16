from __future__ import annotations

from collections.abc import Callable
from typing import AsyncContextManager

import pytest
from mcp.types import Resource

from tests.helpers.mcp_client import (
    MCPClient,
    MCPToolError,
    app_metadata_from_markdown,
    assert_app_home_metadata,
    resource_text,
)


def _pick_home_markdown_resource(resources: list[Resource]) -> Resource:
    markdown_resources = [
        resource
        for resource in resources
        if isinstance(resource.mimeType, str)
        and resource.mimeType.startswith("text/markdown")
    ]
    assert markdown_resources, (
        "Expected at least one markdown app resource from resources/list."
    )

    for resource in markdown_resources:
        uri = str(resource.uri).lower()
        name = resource.name.lower()
        if "home" in uri or "home" in name:
            return resource

    return markdown_resources[0]


async def _read_app_home_metadata(client: MCPClient) -> dict[str, object]:
    resources = await client.list_resources_full()
    home_resource = _pick_home_markdown_resource(resources)
    read_result = await client.read_resource(home_resource.uri)
    return app_metadata_from_markdown(resource_text(read_result))


@pytest.mark.asyncio
async def test_read_profile_excludes_write_tools(
    read_client_factory: Callable[[], AsyncContextManager[MCPClient]],
) -> None:
    async with read_client_factory() as client:
        tools = await client.list_tools()
        metadata = await _read_app_home_metadata(client)
        assert "build" not in tools
        assert "multi_attempt" not in tools
        assert "run_code" not in tools
        assert "outline" in tools
        assert "search" in tools
        assert_app_home_metadata(
            metadata,
            expected_profile="read",
            expected_auth_mode="none",
        )


@pytest.mark.asyncio
async def test_write_profile_includes_write_tools(
    mcp_client_factory: Callable[[], AsyncContextManager[MCPClient]],
) -> None:
    async with mcp_client_factory() as client:
        tools = await client.list_tools()
        metadata = await _read_app_home_metadata(client)
        assert "build" in tools
        assert "multi_attempt" in tools
        assert "run_code" in tools
        assert_app_home_metadata(
            metadata,
            expected_profile="write",
            expected_auth_mode="none",
        )


@pytest.mark.asyncio
async def test_read_profile_rejects_build_call(
    read_client_factory: Callable[[], AsyncContextManager[MCPClient]],
) -> None:
    async with read_client_factory() as client:
        with pytest.raises(MCPToolError):
            await client.call_tool("build", {"clean": False, "output_lines": 5})
