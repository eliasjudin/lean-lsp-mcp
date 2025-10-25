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
            "description": "Retrieve Lean hover documentation for the symbol under the cursor together with `structured.infoSnippet` and nearby diagnostics.",
            "response": "Hover",
            "model": LeanHoverInput,
            "annotations": TOOL_ANNOTATIONS["lean_hover_info"],
            "examples": [
                {"params": {"file_path": "/abs/path/Main.lean", "line": 12, "column": 8, "_format": None}},
                {"params": {"uri": "file:///abs/path/Main.lean", "line": 12, "column": 8, "_format": None}},
            ],
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
            "examples": [
                {"params": {"file_path": "/abs/path/Main.lean", "line": 20, "column": 10, "max_completions": 16, "_format": None}},
                {"params": {"uri": "file:///abs/path/Main.lean", "line": 20, "column": 10, "max_completions": 16, "_format": None}},
            ],
        },
    ),
]

MULTI_TOOLS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "lean_multi_attempt",
        {
            "description": "Evaluate multiple Lean snippets at a line and compare diagnostics/goals.",
            "response": "MultiAttempt",
            "model": LeanMultiAttemptInput,
            "annotations": TOOL_ANNOTATIONS["lean_multi_attempt"],
            "examples": [
                {"params": {"file_path": "/abs/path/Main.lean", "line": 30, "snippets": ["by exact ?_"], "_format": None}},
                {"params": {"uri": "file:///abs/path/Main.lean", "line": 30, "snippets": ["by exact ?_"], "_format": None}},
            ],
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
