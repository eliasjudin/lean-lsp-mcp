from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .annotations import TOOL_ANNOTATIONS
from .build import BUILD_TOOLS
from .diagnostics import DIAGNOSTIC_TOOLS
from .editor import COMPLETION_TOOLS, HOVER_TOOLS, MULTI_TOOLS, RUN_TOOLS
from .files import DECLARATION_TOOLS, FILE_CONTENT_TOOLS
from .goals import GOAL_TOOLS, STATE_TOOLS
from .search import SEARCH_TOOLS
from .spec import SPEC_TOOLS

TOOL_DEFINITIONS: List[Tuple[str, Dict[str, Any]]] = [
    *BUILD_TOOLS,
    *FILE_CONTENT_TOOLS,
    *DIAGNOSTIC_TOOLS,
    *GOAL_TOOLS,
    *HOVER_TOOLS,
    *COMPLETION_TOOLS,
    *DECLARATION_TOOLS,
    *MULTI_TOOLS,
    *RUN_TOOLS,
    *SPEC_TOOLS,
    *SEARCH_TOOLS,
    *STATE_TOOLS,
]

__all__ = ["TOOL_DEFINITIONS", "TOOL_ANNOTATIONS"]
