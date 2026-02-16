from __future__ import annotations

from collections.abc import Callable
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient


def _assert_object_schemas_forbid_extra_fields(tools) -> None:
    payloads = {tool.name: tool.model_dump(by_alias=True) for tool in tools}
    for name, payload in payloads.items():
        schema = payload.get("inputSchema")
        if isinstance(schema, dict) and schema.get("type") == "object":
            assert schema.get("additionalProperties") is False, (
                f"Tool '{name}' must set inputSchema.additionalProperties=false."
            )


@pytest.mark.asyncio
async def test_write_profile_tools_have_strict_object_input_schemas(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    async with remote_client_factory(
        transport="streamable-http", profile="write"
    ) as client:
        tools = await client.list_tools_full()
    _assert_object_schemas_forbid_extra_fields(tools)


@pytest.mark.asyncio
async def test_read_profile_tools_have_strict_object_input_schemas(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> None:
    async with remote_client_factory(
        transport="streamable-http", profile="read"
    ) as client:
        tools = await client.list_tools_full()
    _assert_object_schemas_forbid_extra_fields(tools)
