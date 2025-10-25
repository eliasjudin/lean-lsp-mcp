from __future__ import annotations

from typing import Any, Dict, List, Tuple

from lean_lsp_mcp.tool_inputs import LeanDeclarationFileInput, LeanFileContentsInput

from .annotations import TOOL_ANNOTATIONS

FILE_CONTENT_TOOLS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "lean_file_contents",
        {
            "description": "Read Lean source text with optional 1-indexed annotations; `structured.lines` preserves numbering for downstream agents.",
            "response": "FileContents",
            "model": LeanFileContentsInput,
            "annotations": TOOL_ANNOTATIONS["lean_file_contents"],
            "examples": [
                {"params": {"file_path": "src/Example.lean"}},
                {"params": {"uri": "file:///abs/path/to/Example.lean"}},
            ],
        },
    ),
]

DECLARATION_TOOLS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "lean_declaration_file",
        {
            "description": "Open the Lean source file that defines a given declaration and surface declaration path metadata for navigation.",
            "response": "Declaration",
            "model": LeanDeclarationFileInput,
            "annotations": TOOL_ANNOTATIONS["lean_declaration_file"],
            "examples": [
                {"params": {"file_path": "src/Main.lean", "symbol": "Nat.succ"}},
                {"params": {"uri": "file:///abs/path/to/Main.lean", "symbol": "Nat.succ"}},
            ],
        },
    ),
]

__all__ = ["FILE_CONTENT_TOOLS", "DECLARATION_TOOLS"]
