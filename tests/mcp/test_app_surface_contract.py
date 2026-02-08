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
async def test_app_surface_lists_resources_and_templates(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    async with remote_client_factory(
        transport="streamable-http",
        profile="write",
        auth_mode="none",
    ) as client:
        resources = await client.list_resources_full()
        templates = await client.list_resource_templates_full()

    assert resources, "Expected app resources from resources/list."
    assert templates, "Expected app resource templates from resources/templates/list."
    _pick_home_markdown_resource(resources)

    for template in templates:
        assert template.name, "Resource template name must be non-empty."
        assert template.uriTemplate, "Resource template URI must be non-empty."


@pytest.mark.asyncio
async def test_app_surface_home_resource_contains_markdown_and_metadata(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
    test_project_path: Path,
) -> None:
    async with remote_client_factory(
        transport="streamable-http",
        profile="write",
        auth_mode="none",
    ) as client:
        resources = await client.list_resources_full()
        templates = await client.list_resource_templates_full()
        home_resource = _pick_home_markdown_resource(resources)
        read_result = await client.read_resource(home_resource.uri)

    markdown = resource_text(read_result)
    assert "\n# " in f"\n{markdown}" or "\n## " in f"\n{markdown}", (
        "Expected markdown heading(s) in app home resource."
    )
    metadata = app_metadata_from_markdown(markdown)

    assert_app_home_metadata(
        metadata,
        expected_profile="write",
        expected_auth_mode="none",
        expected_workspace_root=str(test_project_path.resolve()),
    )
    template_uris = {template.uriTemplate for template in templates}
    assert metadata["template_uri"] in template_uris, (
        "App metadata template_uri must match one of resources/templates/list values."
    )
