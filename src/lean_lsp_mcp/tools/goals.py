from __future__ import annotations

from typing import Any, Dict, List, Tuple

from lean_lsp_mcp.tool_inputs import LeanGoalInput, LeanTermGoalInput

from .annotations import TOOL_ANNOTATIONS

GOAL_TOOLS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "lean_goal",
        {
            "description": (
                "Inspect Lean proof goals at a 1-indexed file position; omitting the column probes "
                "line boundaries and returns compact `goals` entries with first-line previews."
            ),
            "response": "Goal",
            "model": LeanGoalInput,
            "annotations": TOOL_ANNOTATIONS["lean_goal"],
        },
    ),
]

STATE_TOOLS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "lean_term_goal",
        {
            "description": (
                "Inspect the expected Lean term type at a 1-indexed location and expose the cleaned "
                "`rendered` snippet alongside normalized coordinates."
            ),
            "response": "Goal",
            "model": LeanTermGoalInput,
            "annotations": TOOL_ANNOTATIONS["lean_term_goal"],
        },
    ),
]

__all__ = ["GOAL_TOOLS", "STATE_TOOLS"]

