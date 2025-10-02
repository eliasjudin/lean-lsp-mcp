"""Helpers for constructing MCP-native tool results."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, MutableMapping

ContentItem = Mapping[str, Any]
StructuredContent = Mapping[str, Any]
MetaMapping = Mapping[str, Any] | MutableMapping[str, Any]


def mcp_result(
    *,
    content: Iterable[ContentItem],
    structured: StructuredContent | None = None,
    is_error: bool = False,
    meta: MetaMapping | None = None,
) -> dict[str, Any]:
    """Return a CallToolResult-compatible payload.

    Parameters
    ----------
    content:
        Iterable of MCP content blocks. The helper copies the items into a list so
        callers can pass generators without leaking iterators.
    structured:
        Optional machine-readable payload exposed via ``structuredContent``.
    is_error:
        Whether the tool execution failed. When ``True`` callers should include a
        descriptive human message as the first content item.
    meta:
        Optional metadata bag replicated into ``_meta`` when provided.
    """

    content_list = list(content)
    if not content_list:
        raise ValueError("mcp_result requires at least one content item")

    result: dict[str, Any] = {
        "content": content_list,
        "isError": bool(is_error),
    }

    if structured is not None:
        result["structuredContent"] = dict(structured)

    if meta:
        result["_meta"] = dict(meta)

    return result


__all__ = ["mcp_result"]
