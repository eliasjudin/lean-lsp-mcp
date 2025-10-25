from __future__ import annotations

from typing import Any, Dict, List, Tuple

from lean_lsp_mcp.tool_inputs import LeanDiagnosticMessagesInput

from .annotations import TOOL_ANNOTATIONS

DIAGNOSTIC_TOOLS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "lean_diagnostic_messages",
        {
            "description": "List Lean diagnostics for a file window; `structured.summary.count` reports totals and `structured.diags` encode zero-based ranges for tooling.",
            "response": "Diagnostics",
            "model": LeanDiagnosticMessagesInput,
            "annotations": TOOL_ANNOTATIONS["lean_diagnostic_messages"],
            # Provide concrete examples to guide agents/UIs.
            "examples": [
                {"params": {"file_path": "src/Example.lean"}},
                {"params": {"uri": "file:///abs/path/to/Example.lean"}},
            ],
        },
    ),
]

__all__ = ["DIAGNOSTIC_TOOLS"]
