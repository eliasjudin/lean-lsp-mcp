from __future__ import annotations

from typing import Any, Dict, List, Tuple

from lean_lsp_mcp.tool_inputs import (
    LeanCompletionsInput,
    LeanHoverInput,
    LeanMultiAttemptInput,
    LeanRunCodeInput,
)

from .annotations import TOOL_ANNOTATIONS

HOVER_TOOLS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "lean_hover_info",
        {
            "description": (
                "Retrieve Lean hover documentation for the symbol under the cursor together with "
                "`structured.infoSnippet` and nearby diagnostics."
            ),
            "response": "Hover",
            "model": LeanHoverInput,
            "annotations": TOOL_ANNOTATIONS["lean_hover_info"],
        },
    ),
]

COMPLETION_TOOLS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "lean_completions",
        {
            "description": "Request Lean completion suggestions at a file position.",
            "response": "Completions",
            "model": LeanCompletionsInput,
            "annotations": TOOL_ANNOTATIONS["lean_completions"],
        },
    ),
]

MULTI_TOOLS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "lean_multi_attempt",
        {
            "description": (
                "Evaluate multiple Lean snippets at a line and compare diagnostics/goals."
            ),
            "response": "MultiAttempt",
            "model": LeanMultiAttemptInput,
            "annotations": TOOL_ANNOTATIONS["lean_multi_attempt"],
        },
    ),
]

RUN_TOOLS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "lean_run_code",
        {
            "description": "Execute an isolated Lean snippet with standalone diagnostics.",
            "response": "RunCode",
            "model": LeanRunCodeInput,
            "annotations": TOOL_ANNOTATIONS["lean_run_code"],
        },
    ),
]

__all__ = ["HOVER_TOOLS", "COMPLETION_TOOLS", "MULTI_TOOLS", "RUN_TOOLS"]

