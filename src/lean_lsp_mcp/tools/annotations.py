from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mcp.types import ToolAnnotations
else:  # pragma: no cover
    try:
        from mcp.types import ToolAnnotations
    except ModuleNotFoundError:
        class ToolAnnotations(dict):
            @classmethod
            def model_validate(cls, values: Dict[str, Any]) -> "ToolAnnotations":
                return cls(values)

            def model_dump(self, *, exclude_none: bool = False) -> Dict[str, Any]:
                return dict(self)

_RAW_TOOL_ANNOTATIONS: Dict[str, Dict[str, Any]] = {
    "lean_build": {
        "title": "Rebuild Lean Project",
        "readOnlyHint": False,
        "idempotentHint": True,
        "destructiveHint": True,
        "openWorldHint": False,
    },
    "lean_file_contents": {
        "title": "Read Lean File",
        "readOnlyHint": True,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": False,
    },
    "lean_diagnostic_messages": {
        "title": "List Lean Diagnostics",
        "readOnlyHint": True,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": False,
    },
    "lean_goal": {
        "title": "Inspect Lean Goals",
        "readOnlyHint": True,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": False,
    },
    "lean_term_goal": {
        "title": "Inspect Expected Term Type",
        "readOnlyHint": True,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": False,
    },
    "lean_hover_info": {
        "title": "Inspect Hover Info",
        "readOnlyHint": True,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": False,
    },
    "lean_completions": {
        "title": "Request Completions",
        "readOnlyHint": True,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": False,
    },
    "lean_declaration_file": {
        "title": "Open Declaration File",
        "readOnlyHint": True,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": False,
    },
    "lean_multi_attempt": {
        "title": "Compare Snippet Attempts",
        "readOnlyHint": False,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": False,
    },
    "lean_run_code": {
        "title": "Run Isolated Snippet",
        "readOnlyHint": False,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": False,
    },
    "lean_tool_spec": {
        "title": "Export Tool Specification",
        "readOnlyHint": True,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": False,
    },
    "lean_leansearch": {
        "title": "Search leansearch.net",
        "readOnlyHint": True,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    },
    "lean_loogle": {
        "title": "Search Loogle",
        "readOnlyHint": True,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    },
    "lean_state_search": {
        "title": "Goal-Based Lemma Search",
        "readOnlyHint": True,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    },
    "lean_hammer_premise": {
        "title": "Fetch Hammer Premises",
        "readOnlyHint": True,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    },
}

TOOL_ANNOTATIONS: Dict[str, ToolAnnotations] = {
    name: ToolAnnotations.model_validate(values)
    for name, values in _RAW_TOOL_ANNOTATIONS.items()
}

__all__ = ["TOOL_ANNOTATIONS"]
