from __future__ import annotations

from typing import Any, Dict, List, Tuple

from lean_lsp_mcp.tool_inputs import (
    LeanGoalInput,
    LeanHammerPremiseInput,
    LeanStateSearchInput,
    LeanTermGoalInput,
)

from .annotations import TOOL_ANNOTATIONS

GOAL_TOOLS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "lean_goal",
        {
            "description": "Inspect Lean proof goals at a 1-indexed file position; omitting the column probes line boundaries and returns compact `goals` entries with first-line previews.",
            "response": "Goal",
            "model": LeanGoalInput,
            "annotations": TOOL_ANNOTATIONS["lean_goal"],
            "examples": [
                {"params": {"file_path": "/abs/path/Main.lean", "line": 15, "column": 5, "_format": None}},
                {"params": {"uri": "file:///abs/path/Main.lean", "line": 15, "column": 5, "_format": None}},
            ],
        },
    ),
    (
        "lean_term_goal",
        {
            "description": "Inspect the expected Lean term type at a 1-indexed location and expose the cleaned `rendered` snippet alongside normalized coordinates.",
            "response": "Goal",
            "model": LeanTermGoalInput,
            "annotations": TOOL_ANNOTATIONS["lean_term_goal"],
            "examples": [
                {"params": {"file_path": "/abs/path/Main.lean", "line": 15, "column": 5, "_format": None}},
                {"params": {"uri": "file:///abs/path/Main.lean", "line": 15, "column": 5, "_format": None}},
            ],
        },
    ),
]

STATE_TOOLS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "lean_state_search",
        {
            "description": "Fetch goal-based lemma suggestions from premise-search.com (3 req/30s).",
            "response": "SearchResults",
            "model": LeanStateSearchInput,
            "rate_limit": {"category": "lean_state_search", "max": 3, "window_seconds": 30},
            "annotations": TOOL_ANNOTATIONS["lean_state_search"],
            "examples": [
                {"params": {"file_path": "/abs/path/Main.lean", "line": 22, "column": 6, "num_results": 5, "_format": None}},
                {"params": {"uri": "file:///abs/path/Main.lean", "line": 22, "column": 6, "num_results": 5, "_format": None}},
            ],
        },
    ),
    (
        "lean_hammer_premise",
        {
            "description": "Retrieve hammer premise suggestions for the active goal (3 req/30s).",
            "response": "SearchResults",
            "model": LeanHammerPremiseInput,
            "rate_limit": {"category": "hammer_premise", "max": 3, "window_seconds": 30},
            "annotations": TOOL_ANNOTATIONS["lean_hammer_premise"],
            "examples": [
                {"params": {"file_path": "/abs/path/Main.lean", "line": 22, "column": 6, "num_results": 16, "_format": None}},
                {"params": {"uri": "file:///abs/path/Main.lean", "line": 22, "column": 6, "num_results": 16, "_format": None}},
            ],
        },
    ),
]

__all__ = ["GOAL_TOOLS", "STATE_TOOLS"]
