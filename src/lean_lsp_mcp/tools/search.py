from __future__ import annotations

from typing import Any, Dict, List, Tuple

from lean_lsp_mcp.tool_inputs import (
    LeanHammerPremiseInput,
    LeanSearchInput,
    LeanStateSearchInput,
    LoogleSearchInput,
)

from .annotations import TOOL_ANNOTATIONS

SEARCH_TOOLS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "lean_leansearch",
        {
            "description": "Query leansearch.net for Lean lemmas/theorems; responses list fully qualified names in `structured.names` (3 req/30s rate limit).",
            "response": "SearchResults",
            "model": LeanSearchInput,
            "rate_limit": {"category": "leansearch", "max": 3, "window_seconds": 30},
            "annotations": TOOL_ANNOTATIONS["lean_leansearch"],
        },
    ),
    (
        "lean_loogle",
        {
            "description": "Query loogle.lean-lang.org for Lean declarations by name/pattern; compact responses populate `structured.names` (3 req/30s).",
            "response": "SearchResults",
            "model": LoogleSearchInput,
            "rate_limit": {"category": "loogle", "max": 3, "window_seconds": 30},
            "annotations": TOOL_ANNOTATIONS["lean_loogle"],
        },
    ),
    (
        "lean_state_search",
        {
            "description": "Fetch goal-based lemma suggestions from premise-search.com (3 req/30s).",
            "response": "SearchResults",
            "model": LeanStateSearchInput,
            "rate_limit": {
                "category": "lean_state_search",
                "max": 3,
                "window_seconds": 30,
            },
            "annotations": TOOL_ANNOTATIONS["lean_state_search"],
        },
    ),
    (
        "lean_hammer_premise",
        {
            "description": "Retrieve hammer premise suggestions for the active goal (3 req/30s).",
            "response": "SearchResults",
            "model": LeanHammerPremiseInput,
            "rate_limit": {
                "category": "hammer_premise",
                "max": 3,
                "window_seconds": 30,
            },
            "annotations": TOOL_ANNOTATIONS["lean_hammer_premise"],
        },
    ),
]

__all__ = ["SEARCH_TOOLS"]
