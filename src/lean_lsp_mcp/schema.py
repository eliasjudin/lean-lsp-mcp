"""Helpers for constructing MCP-native tool results."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

ContentItem = Mapping[str, Any]
StructuredContent = Mapping[str, Any]


def mcp_result(
    *,
    content: Iterable[ContentItem],
    structured: StructuredContent | None = None,
    is_error: bool = False,
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
    """

    content_list = []
    for index, item in enumerate(content):
        if not isinstance(item, Mapping):
            raise TypeError(
                f"content item at index {index} must be a Mapping, "
                f"got {type(item).__name__}"
            )
        content_list.append(dict(item))
    if not content_list:
        raise ValueError("mcp_result requires at least one content item")

    result: dict[str, Any] = {
        "content": content_list,
        "isError": bool(is_error),
    }

    if structured is not None:
        if not isinstance(structured, Mapping):
            raise TypeError(
                "structured must be a Mapping when provided"
            )

        structured_dict = dict(structured)
        try:
            json.dumps(structured_dict)
        except TypeError as exc:
            raise TypeError("structured must be JSON-serializable") from exc

        result["structuredContent"] = structured_dict

    return result


__all__ = ["mcp_result"]
