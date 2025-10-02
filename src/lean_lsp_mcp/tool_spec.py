"""Lightweight machine-readable tool specification."""

from __future__ import annotations

from typing import Any, Dict, List

TOOL_SPEC_VERSION = "2025-10-02"

BASIC_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "lean_build",
        "description": "Build the Lean project and restart the LSP server.",
        "inputs": [
            {"name": "lean_project_path", "type": "string", "required": False},
            {"name": "clean", "type": "boolean", "required": False, "default": False},
            {"name": "format", "type": "string", "required": False, "enum": ["compact", "verbose"], "default": "compact"},
        ],
        "response": "LeanBuildResult",
    },
    {
        "name": "lean_file_contents",
        "description": "Get the content of a Lean file.",
        "inputs": [
            {"name": "file_path", "type": "string", "required": True},
            {"name": "annotate_lines", "type": "boolean", "required": False, "default": True},
            {"name": "start_line", "type": "integer", "required": False},
            {"name": "line_count", "type": "integer", "required": False},
            {"name": "format", "type": "string", "required": False, "enum": ["compact", "verbose"], "default": "compact"},
        ],
        "response": "FileContents",
    },
    {
        "name": "lean_diagnostic_messages",
        "description": "Fetch diagnostics for a Lean file.",
        "inputs": [
            {"name": "file_path", "type": "string", "required": True},
            {"name": "start_line", "type": "integer", "required": False},
            {"name": "line_count", "type": "integer", "required": False},
            {
                "name": "format",
                "type": "string",
                "required": False,
                "enum": ["compact", "verbose"],
                "default": "compact"
            },
        ],
        "response": "Diagnostics",
    },
    {
        "name": "lean_goal",
        "description": "Return the goals at a given location.",
        "inputs": [
            {"name": "file_path", "type": "string", "required": True},
            {"name": "line", "type": "integer", "required": True},
            {"name": "column", "type": "integer", "required": False},
            {
                "name": "format",
                "type": "string",
                "required": False,
                "enum": ["compact", "verbose"],
                "default": "compact"
            },
        ],
        "response": "Goal",
    },
    {
        "name": "lean_term_goal",
        "description": "Return the expected term type at a location.",
        "inputs": [
            {"name": "file_path", "type": "string", "required": True},
            {"name": "line", "type": "integer", "required": True},
            {"name": "column", "type": "integer", "required": False},
            {"name": "format", "type": "string", "required": False, "enum": ["compact", "verbose"], "default": "compact"},
        ],
        "response": "Goal",
    },
    {
        "name": "lean_hover_info",
        "description": "Return hover info for a position.",
        "inputs": [
            {"name": "file_path", "type": "string", "required": True},
            {"name": "line", "type": "integer", "required": True},
            {"name": "column", "type": "integer", "required": True},
            {"name": "format", "type": "string", "required": False, "enum": ["compact", "verbose"], "default": "compact"},
        ],
        "response": "Hover",
    },
    {
        "name": "lean_completions",
        "description": "Return completion suggestions at a position.",
        "inputs": [
            {"name": "file_path", "type": "string", "required": True},
            {"name": "line", "type": "integer", "required": True},
            {"name": "column", "type": "integer", "required": True},
            {"name": "max_completions", "type": "integer", "required": False, "default": 32},
            {"name": "format", "type": "string", "required": False, "enum": ["compact", "verbose"], "default": "compact"},
        ],
        "response": "Completions",
    },
    {
        "name": "lean_declaration_file",
        "description": "Return the declaration file for a symbol.",
        "inputs": [
            {"name": "file_path", "type": "string", "required": True},
            {"name": "symbol", "type": "string", "required": True},
            {"name": "format", "type": "string", "required": False, "enum": ["compact", "verbose"], "default": "compact"},
        ],
        "response": "Declaration",
    },
    {
        "name": "lean_multi_attempt",
        "description": "Apply multiple snippets at a line and compare results.",
        "inputs": [
            {"name": "file_path", "type": "string", "required": True},
            {"name": "line", "type": "integer", "required": True},
            {"name": "snippets", "type": "array", "items": {"type": "string"}, "required": True},
            {"name": "format", "type": "string", "required": False, "enum": ["compact", "verbose"], "default": "compact"},
        ],
        "response": "MultiAttempt",
    },
    {
        "name": "lean_run_code",
        "description": "Run an isolated Lean snippet and report diagnostics.",
        "inputs": [
            {"name": "code", "type": "string", "required": True},
            {"name": "format", "type": "string", "required": False, "enum": ["compact", "verbose"], "default": "compact"}
        ],
        "response": "RunCode",
    },
    {
        "name": "lean_leansearch",
        "description": "Search Lean theorems via leansearch.net.",
        "inputs": [
            {"name": "query", "type": "string", "required": True},
            {"name": "num_results", "type": "integer", "required": False, "default": 5},
            {"name": "format", "type": "string", "required": False, "enum": ["compact", "verbose"], "default": "compact"},
        ],
        "response": "SearchResults",
        "rate_limit": {"category": "leansearch", "max": 3, "window_seconds": 30},
    },
    {
        "name": "lean_loogle",
        "description": "Search definitions and theorems via loogle.",
        "inputs": [
            {"name": "query", "type": "string", "required": True},
            {"name": "num_results", "type": "integer", "required": False, "default": 8},
            {"name": "format", "type": "string", "required": False, "enum": ["compact", "verbose"], "default": "compact"},
        ],
        "response": "SearchResults",
        "rate_limit": {"category": "loogle", "max": 3, "window_seconds": 30},
    },
    {
        "name": "lean_state_search",
        "description": "Goal-directed search using premise-search.com.",
        "inputs": [
            {"name": "file_path", "type": "string", "required": True},
            {"name": "line", "type": "integer", "required": True},
            {"name": "column", "type": "integer", "required": True},
            {"name": "num_results", "type": "integer", "required": False, "default": 5},
            {"name": "format", "type": "string", "required": False, "enum": ["compact", "verbose"], "default": "compact"},
        ],
        "response": "SearchResults",
        "rate_limit": {"category": "lean_state_search", "max": 3, "window_seconds": 30},
    },
    {
        "name": "lean_hammer_premise",
        "description": "Fetch premises via the Lean hammer service.",
        "inputs": [
            {"name": "file_path", "type": "string", "required": True},
            {"name": "line", "type": "integer", "required": True},
            {"name": "column", "type": "integer", "required": True},
            {"name": "num_results", "type": "integer", "required": False, "default": 32},
            {"name": "format", "type": "string", "required": False, "enum": ["compact", "verbose"], "default": "compact"},
        ],
        "response": "SearchResults",
        "rate_limit": {"category": "hammer_premise", "max": 3, "window_seconds": 30},
    },
]

RESPONSE_KIND_SUMMARY: Dict[str, Dict[str, Any]] = {
    "LeanBuildResult": {"description": "Build output and project path."},
    "FileContents": {"description": "File contents with optional line annotations."},
    "Diagnostics": {"description": "Diagnostics emitted by the Lean LSP."},
    "Goal": {"description": "Goal state at a location."},
    "Hover": {"description": "Hover information for a symbol."},
    "Completions": {"description": "Completion suggestions."},
    "Declaration": {"description": "Declaration text for a symbol."},
    "MultiAttempt": {"description": "Diagnostics and goals per snippet."},
    "RunCode": {"description": "Diagnostics for a temporary snippet."},
    "SearchResults": {"description": "Results from external search services."},
}


def build_tool_spec() -> Dict[str, Any]:
    return {
        "version": TOOL_SPEC_VERSION,
        "tools": BASIC_TOOLS,
        "responses": RESPONSE_KIND_SUMMARY,
    }


__all__ = ["build_tool_spec", "TOOL_SPEC_VERSION"]
