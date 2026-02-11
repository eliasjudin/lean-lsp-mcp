from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, Mapping

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

try:
    from mcp.client.streamable_http import streamable_http_client
except ImportError:  # mcp<1.26 compatibility
    streamable_http_client = None  # type: ignore[assignment]
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    ReadResourceResult,
    Resource,
    ResourceLink,
    ResourceTemplate,
    TextContent,
    TextResourceContents,
)
from pydantic import AnyUrl

APP_METADATA_REQUIRED_KEYS = {
    "app_name",
    "profile",
    "auth_mode",
    "transport",
    "workspace_root",
    "template_uri",
    "tool_groups",
}

APP_METADATA_TOOL_GROUP_KEYS = {
    "read",
    "write",
    "enabled",
}

APP_METADATA_TRANSPORT_KEYS = {
    "streamable_http",
    "sse",
}


class MCPClientError(RuntimeError):
    """Base exception raised for MCP helper failures."""


@dataclass(slots=True)
class MCPToolError(MCPClientError):
    """Exception raised when a tool returns an error payload."""

    tool_name: str
    result: CallToolResult

    def __str__(self) -> str:  # pragma: no cover - trivial string repr
        return f"Tool '{self.tool_name}' failed: {result_text(self.result)}"


class MCPClient:
    """Lightweight helper around :class:`mcp.client.session.ClientSession`."""

    def __init__(self, session: ClientSession, *, endpoint: str | None = None) -> None:
        self._session = session
        self.endpoint = endpoint
        if endpoint and endpoint.endswith("/mcp"):
            self.base_url = endpoint[: -len("/mcp")]
        elif endpoint and endpoint.endswith("/sse"):
            self.base_url = endpoint[: -len("/sse")]
        else:
            self.base_url = endpoint

    async def list_tools(self) -> list[str]:
        result = await self._session.list_tools()
        return [tool.name for tool in result.tools]

    async def list_tools_full(self):
        result = await self._session.list_tools()
        return list(result.tools)

    async def list_resources(self) -> list[str]:
        result = await self._session.list_resources()
        return [str(resource.uri) for resource in result.resources]

    async def list_resources_full(self) -> list[Resource]:
        result = await self._session.list_resources()
        return list(result.resources)

    async def list_resource_templates(self) -> list[str]:
        result = await self._session.list_resource_templates()
        return [template.uriTemplate for template in result.resourceTemplates]

    async def list_resource_templates_full(self) -> list[ResourceTemplate]:
        result = await self._session.list_resource_templates()
        return list(result.resourceTemplates)

    async def read_resource(self, uri: str | AnyUrl) -> ReadResourceResult:
        return await self._session.read_resource(uri)  # type: ignore[arg-type]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        expect_error: bool = False,
    ) -> CallToolResult:
        result = await self._session.call_tool(name, arguments or {})
        if result.isError and not expect_error:
            raise MCPToolError(name, result)
        return result


@asynccontextmanager
async def connect_streamable_http_client(
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> AsyncIterator[MCPClient]:
    if streamable_http_client is None:
        async with streamablehttp_client(url, headers=headers) as (
            read_stream,
            write_stream,
            _,
        ):
            session = ClientSession(read_stream, write_stream)
            async with session:
                await session.initialize()
                yield MCPClient(session, endpoint=url)
        return

    http_client: httpx.AsyncClient | None = None
    if headers:
        http_client = httpx.AsyncClient(headers=headers)
    try:
        async with streamable_http_client(url, http_client=http_client) as (
            read_stream,
            write_stream,
            _,
        ):
            session = ClientSession(read_stream, write_stream)
            async with session:
                await session.initialize()
                yield MCPClient(session, endpoint=url)
    finally:
        if http_client is not None:
            await http_client.aclose()


@asynccontextmanager
async def connect_sse_client(
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> AsyncIterator[MCPClient]:
    async with sse_client(url, headers=headers) as (read_stream, write_stream):
        session = ClientSession(read_stream, write_stream)
        async with session:
            await session.initialize()
            yield MCPClient(session, endpoint=url)


def result_text(result: CallToolResult) -> str:
    segments: list[str] = []
    for block in result.content:
        segments.extend(_text_from_block(block))
    return "\n".join(segment for segment in segments if segment)


def result_json(result: CallToolResult) -> dict[str, Any]:
    return json.loads(result_text(result))


def resource_text(result: ReadResourceResult) -> str:
    segments: list[str] = []
    for content in result.contents:
        segments.extend(_text_from_resource_content(content))
    return "\n".join(segment for segment in segments if segment)


def app_metadata_from_markdown(markdown: str) -> dict[str, Any]:
    fenced_blocks = re.finditer(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        markdown,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for block in fenced_blocks:
        payload = block.group(1)
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            return decoded

    stripped = markdown.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        decoded = json.loads(stripped)
        if isinstance(decoded, dict):
            return decoded

    raise MCPClientError(
        "Expected app metadata as a JSON object inside markdown (for example, in a fenced json block)."
    )


def assert_app_home_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_profile: str,
    expected_auth_mode: str,
    expected_workspace_root: str | None = None,
) -> None:
    missing_top_level = APP_METADATA_REQUIRED_KEYS.difference(metadata.keys())
    assert not missing_top_level, (
        f"Missing app metadata keys: {sorted(missing_top_level)}"
    )

    app_name = metadata["app_name"]
    assert isinstance(app_name, str) and app_name.strip(), (
        "app_name must be a non-empty string."
    )
    assert metadata["profile"] == expected_profile, (
        f"Expected profile '{expected_profile}', got '{metadata['profile']}'."
    )
    assert metadata["auth_mode"] == expected_auth_mode, (
        f"Expected auth_mode '{expected_auth_mode}', got '{metadata['auth_mode']}'."
    )

    workspace_root = metadata["workspace_root"]
    assert isinstance(workspace_root, str) and workspace_root, (
        "workspace_root must be a non-empty string."
    )
    if expected_workspace_root is not None:
        assert workspace_root == expected_workspace_root, (
            f"Expected workspace_root '{expected_workspace_root}', got '{workspace_root}'."
        )

    template_uri = metadata["template_uri"]
    assert isinstance(template_uri, str) and template_uri, (
        "template_uri must be a non-empty string."
    )

    transport_paths = metadata.get("transport_paths")
    if transport_paths is not None:
        assert isinstance(transport_paths, dict), "transport_paths must be an object."
        missing_transport = APP_METADATA_TRANSPORT_KEYS.difference(
            transport_paths.keys()
        )
        assert not missing_transport, (
            f"Missing transport path keys: {sorted(missing_transport)}"
        )
        assert transport_paths["streamable_http"] == "/mcp", (
            "transport_paths.streamable_http must be '/mcp'."
        )
        assert transport_paths["sse"] == "/sse", "transport_paths.sse must be '/sse'."

    tool_groups = metadata["tool_groups"]
    assert isinstance(tool_groups, dict), "tool_groups must be an object."
    missing_tool_groups = APP_METADATA_TOOL_GROUP_KEYS.difference(tool_groups.keys())
    assert not missing_tool_groups, (
        f"Missing tool_groups keys: {sorted(missing_tool_groups)}"
    )

    read_tools = _tool_name_set(tool_groups["read"], field="tool_groups.read")
    write_tools = _tool_name_set(tool_groups["write"], field="tool_groups.write")
    enabled_tools = _tool_name_set(tool_groups["enabled"], field="tool_groups.enabled")

    assert "search" in read_tools, "tool_groups.read must include search."
    assert "build" in write_tools, "tool_groups.write must include build."
    assert read_tools.isdisjoint(write_tools), (
        "tool_groups.read and tool_groups.write must be disjoint."
    )
    assert enabled_tools.issubset(read_tools | write_tools), (
        "tool_groups.enabled must be a subset of read∪write tools."
    )
    if expected_profile == "read":
        assert write_tools.isdisjoint(enabled_tools), (
            "Read profile must not enable write tools."
        )
    elif expected_profile == "write":
        assert write_tools.issubset(enabled_tools), (
            "Write profile must enable all write tools."
        )


def _text_from_block(block: ContentBlock) -> Iterable[str]:
    if isinstance(block, TextContent):
        yield block.text
    elif isinstance(block, ResourceLink):
        yield block.uri
    elif isinstance(block, EmbeddedResource):
        resource = block.resource
        if hasattr(resource, "text"):
            yield resource.text  # type: ignore[no-any-return]
    else:
        yield ""


def _text_from_resource_content(
    content: TextResourceContents | BlobResourceContents,
) -> Iterable[str]:
    if isinstance(content, TextResourceContents):
        yield content.text
    elif isinstance(content, BlobResourceContents):
        yield content.blob


def _tool_name_set(value: Any, *, field: str) -> set[str]:
    assert isinstance(value, list), f"{field} must be a list."
    assert all(isinstance(item, str) and item for item in value), (
        f"{field} entries must be non-empty strings."
    )
    return set(value)
