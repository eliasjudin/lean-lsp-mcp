from __future__ import annotations

from typing import Any, Dict, List, Tuple

from lean_lsp_mcp.tool_inputs import LeanBuildInput

from .annotations import TOOL_ANNOTATIONS

BUILD_TOOLS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "lean_build",
        {
            "description": "Run `lake build` (optionally `lake clean`) to refresh the Lean project and restart the cached Lean LSP client.",
            "response": "LeanBuildResult",
            "model": LeanBuildInput,
            "annotations": TOOL_ANNOTATIONS["lean_build"],
        },
    ),
]

__all__ = ["BUILD_TOOLS"]
