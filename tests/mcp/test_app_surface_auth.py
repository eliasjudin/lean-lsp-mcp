from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest
from mcp.types import Resource

from tests.helpers.mcp_client import (
    MCPClient,
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


@pytest.mark.asyncio
async def test_mixed_mode_allows_unauthenticated_app_surface_discovery(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    async with remote_client_factory(
        transport="streamable-http",
        profile="write",
        auth_mode="mixed",
        token="server-token",
        client_token="",
    ) as client:
        resources = await client.list_resources_full()
        templates = await client.list_resource_templates_full()
        home_resource = _pick_home_markdown_resource(resources)
        read_result = await client.read_resource(home_resource.uri)

    assert templates, "Expected resource templates in mixed auth mode."
    metadata = app_metadata_from_markdown(resource_text(read_result))
    assert_app_home_metadata(
        metadata,
        expected_profile="write",
        expected_auth_mode="mixed",
        expected_workspace_root=str(test_project_path.resolve()),
    )


@pytest.mark.asyncio
async def test_bearer_mode_rejects_invalid_token_for_app_surface(
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
            await client.list_resources_full()


@pytest.mark.asyncio
async def test_bearer_mode_valid_token_reads_app_metadata(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    async with remote_client_factory(
        transport="streamable-http",
        profile="write",
        auth_mode="bearer",
        token="server-token",
        client_token="server-token",
    ) as client:
        resources = await client.list_resources_full()
        home_resource = _pick_home_markdown_resource(resources)
        read_result = await client.read_resource(home_resource.uri)

    metadata = app_metadata_from_markdown(resource_text(read_result))
    assert_app_home_metadata(
        metadata,
        expected_profile="write",
        expected_auth_mode="bearer",
        expected_workspace_root=str(test_project_path.resolve()),
    )
