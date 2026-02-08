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
