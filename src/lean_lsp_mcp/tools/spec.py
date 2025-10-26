from __future__ import annotations

from typing import Any, Dict, List, Tuple

from lean_lsp_mcp.tool_inputs import LeanToolSpecInput

from .annotations import TOOL_ANNOTATIONS

SPEC_TOOLS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "lean_tool_spec",
        {
            "description": "Publish structured metadata for all Lean MCP tools, including inputs, annotations, and rate limits.",
            "response": "ToolSpecSummary",
            "model": LeanToolSpecInput,
            "annotations": TOOL_ANNOTATIONS["lean_tool_spec"],
        },
    ),
]

__all__ = ["SPEC_TOOLS"]
