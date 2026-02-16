from __future__ import annotations

from collections.abc import Callable
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient


REQUIRED_ANNOTATION_KEYS = (
    "readOnlyHint",
    "destructiveHint",
    "idempotentHint",
    "openWorldHint",
)

EXTERNAL_QUERY_TOOLS = (
    "local_search",
    "leansearch",
    "loogle",
    "leanfinder",
    "state_search",
    "hammer_premise",
)


def _assert_tool_annotations(tools) -> None:
    payloads = {
        tool.name: tool.model_dump(by_alias=True, exclude_none=False) for tool in tools
    }
    for name, payload in payloads.items():
        annotations = payload.get("annotations")
        assert isinstance(annotations, dict), (
            f"Tool '{name}' must include annotations metadata."
        )
        for key in REQUIRED_ANNOTATION_KEYS:
            assert isinstance(annotations.get(key), bool), (
                f"Tool '{name}' must include boolean annotations.{key}."
            )


@pytest.mark.asyncio
async def test_read_profile_tools_export_required_annotations(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    async with remote_client_factory(
        transport="streamable-http", profile="read"
    ) as client:
        tools = await client.list_tools_full()
    _assert_tool_annotations(tools)


@pytest.mark.asyncio
async def test_write_profile_tools_export_required_annotations(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    async with remote_client_factory(
        transport="streamable-http", profile="write"
    ) as client:
        tools = await client.list_tools_full()
    _assert_tool_annotations(tools)


@pytest.mark.asyncio
async def test_external_query_tools_are_read_only(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    async with remote_client_factory(
        transport="streamable-http", profile="read"
    ) as client:
        tools = await client.list_tools_full()

    payloads = {
        tool.name: tool.model_dump(by_alias=True, exclude_none=False) for tool in tools
    }
    missing = [name for name in EXTERNAL_QUERY_TOOLS if name not in payloads]
    assert not missing, f"Expected external query tools to be exported: {missing}"

    for name in EXTERNAL_QUERY_TOOLS:
        annotations = payloads[name]["annotations"]
        assert annotations["readOnlyHint"] is True, (
            f"Tool '{name}' should be marked read-only."
        )
