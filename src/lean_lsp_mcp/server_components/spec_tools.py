from __future__ import annotations

import json
import time
from typing import Any, Dict

from mcp.server.fastmcp import Context

from lean_lsp_mcp.tool_inputs import LeanToolSpecInput
from lean_lsp_mcp.tool_spec import TOOL_ANNOTATIONS, TOOL_SPEC_VERSION, build_tool_spec

from .common import (
    _resource_item,
    _set_response_format_hint,
    _text_item,
    success_result,
)
from .context import mcp

TOOL_SPEC_RESOURCE_URI = f"tool-spec://lean_lsp_mcp/{TOOL_SPEC_VERSION}.json"

__all__ = ["tool_spec", "TOOL_SPEC_RESOURCE_URI"]


@mcp.tool(
    "lean_tool_spec",
    description="Publish structured metadata for all Lean MCP tools, including inputs, annotations, and rate limits.",
    annotations=TOOL_ANNOTATIONS["lean_tool_spec"],
)
def tool_spec(ctx: Context, params: LeanToolSpecInput) -> Any:
    """Return the published Lean MCP tool specification for evaluation harnesses.

    Parameters
    ----------
    params : LeanToolSpecInput
        Optional request metadata.
        - ``response_format`` (Optional[str]): Legacy hint accepted for compatibility.

    Returns
    -------
    ToolSpecSummary
        ``structuredContent`` contains the result of
        ``lean_lsp_mcp.tool_spec.build_tool_spec`` including version, tool definitions,
        and response summaries. The human-readable content includes a short summary and
        an attached JSON resource whose URI encodes the spec version for caching.
        For direct module access use
        ``lean_lsp_mcp.tool_spec.build_tool_spec`` inside the Python package.
    """
    response_format = params.response_format
    _set_response_format_hint(ctx, response_format)
    started = time.perf_counter()
    spec, spec_json = _render_tool_spec_payload()
    summary = "Lean MCP tool specification ready."
    content_items = [
        _text_item(summary),
        _resource_item(TOOL_SPEC_RESOURCE_URI, spec_json, mime_type="application/json"),
    ]
    return success_result(
        summary=summary,
        structured=spec,
        content=content_items,
        start_time=started,
        ctx=ctx,
    )


def _render_tool_spec_payload() -> tuple[Dict[str, Any], str]:
    spec = build_tool_spec()
    spec_json = json.dumps(spec, indent=2, ensure_ascii=False)
    return spec, spec_json


def _tool_spec_resource_impl() -> str:
    _, spec_json = _render_tool_spec_payload()
    return spec_json


_resource_reg = getattr(mcp, "resource", None)
if callable(_resource_reg):  # pragma: no cover - exercised in real server runtime
    _tool_spec_resource_impl = _resource_reg(
        TOOL_SPEC_RESOURCE_URI,
        title="Lean MCP Tool Specification",
        description=(
            "Structured metadata for all Lean MCP tools, including schemas and annotations."
        ),
        mime_type="application/json",
    )(_tool_spec_resource_impl)
