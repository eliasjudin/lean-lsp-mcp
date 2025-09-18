"""Machine-readable tool specification for Lean LSP MCP."""

from __future__ import annotations

from typing import Any, Dict, List

from .schema import SCHEMA_VERSION

TOOL_SPEC_VERSION = "2024-06-01"

OUTPUT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "LeanBuildResult": {
        "description": "Result of building a Lean project",
        "properties": {
            "project_path": {"type": "string"},
            "output": {"type": "string"},
            "clean": {"type": "boolean"},
        },
    },
    "FileContents": {
        "description": "File contents payload with optional annotation",
        "properties": {
            "path": {"type": "string"},
            "annotated": {"type": "boolean"},
            "lines": {
                "type": "array",
                "items": {"type": "object", "properties": {"number": {"type": "integer"}, "text": {"type": "string"}}},
            },
            "contents": {"type": "string"},
        },
    },
    "Diagnostics": {
        "description": "Diagnostics for a Lean file",
        "properties": {
            "file": {"type": "string"},
            "diagnostics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "severity": {"type": "integer"},
                        "range": {"type": "object"},
                        "code": {"type": ["string", "integer", "null"]},
                        "source": {"type": ["string", "null"]},
                    },
                },
            },
        },
    },
    "GoalState": {
        "description": "Goal information for a Lean location",
        "properties": {
            "file": {"type": "string"},
            "request": {"type": "object"},
            "line": {"type": "string"},
            "line_with_cursor": {"type": "string"},
            "goal": {"type": ["object", "null"]},
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "position": {"type": "string"},
                        "goal": {"type": ["object", "null"]},
                    },
                },
            },
        },
    },
    "HoverInfo": {
        "description": "Hover information for a Lean symbol",
        "properties": {
            "file": {"type": "string"},
            "position": {"type": "object"},
            "symbol": {"type": "string"},
            "range": {"type": ["object", "null"]},
            "info": {"type": "string"},
            "diagnostics": {"type": "array"},
        },
    },
    "Completions": {
        "description": "Completion suggestions for a Lean cursor position",
        "properties": {
            "file": {"type": "string"},
            "position": {"type": "object"},
            "prefix": {"type": "string"},
            "suggestions": {
                "type": "array",
                "items": {"type": "object"},
            },
            "line": {"type": "string"},
            "line_with_cursor": {"type": "string"},
        },
    },
    "Declaration": {
        "description": "Declaration contents for a symbol",
        "properties": {
            "symbol": {"type": "string"},
            "origin_file": {"type": "string"},
            "declaration": {"type": "object"},
        },
    },
    "MultiAttempt": {
        "description": "Results of applying multiple snippets",
        "properties": {
            "file": {"type": "string"},
            "line": {"type": "integer"},
            "attempts": {
                "type": "array",
                "items": {"type": "object"},
            },
        },
    },
    "RunCode": {
        "description": "Diagnostics for a temporary Lean snippet",
        "properties": {
            "snippet_path": {"type": "string"},
            "diagnostics": {"type": "array"},
        },
    },
    "SearchResults": {
        "description": "Search results from remote services",
        "properties": {
            "query": {"type": "string"},
            "results": {"type": "array"},
            "count": {"type": "integer"},
        },
    },
}

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "lean_build",
        "description": "Build the Lean project and restart the LSP server.",
        "inputs": [
            {"name": "lean_project_path", "type": "string", "required": False},
            {"name": "clean", "type": "boolean", "required": False, "default": False},
        ],
        "output": "LeanBuildResult",
    },
    {
        "name": "lean_file_contents",
        "description": "Get the content of a Lean file, optionally annotated with line numbers.",
        "inputs": [
            {"name": "file_path", "type": "string", "required": True},
            {"name": "annotate_lines", "type": "boolean", "required": False, "default": True},
        ],
        "output": "FileContents",
    },
    {
        "name": "lean_diagnostic_messages",
        "description": "Fetch diagnostics for the specified Lean file.",
        "inputs": [{"name": "file_path", "type": "string", "required": True}],
        "output": "Diagnostics",
    },
    {
        "name": "lean_goal",
        "description": "Return the goals at a given location.",
        "inputs": [
            {"name": "file_path", "type": "string", "required": True},
            {"name": "line", "type": "integer", "required": True},
            {"name": "column", "type": "integer", "required": False},
        ],
        "output": "GoalState",
    },
    {
        "name": "lean_term_goal",
        "description": "Return the expected term type at a location.",
        "inputs": [
            {"name": "file_path", "type": "string", "required": True},
            {"name": "line", "type": "integer", "required": True},
            {"name": "column", "type": "integer", "required": False},
        ],
        "output": "GoalState",
    },
    {
        "name": "lean_hover_info",
        "description": "Return hover info for a position.",
        "inputs": [
            {"name": "file_path", "type": "string", "required": True},
            {"name": "line", "type": "integer", "required": True},
            {"name": "column", "type": "integer", "required": True},
        ],
        "output": "HoverInfo",
    },
    {
        "name": "lean_completions",
        "description": "Return completion suggestions at a position.",
        "inputs": [
            {"name": "file_path", "type": "string", "required": True},
            {"name": "line", "type": "integer", "required": True},
            {"name": "column", "type": "integer", "required": True},
            {"name": "max_completions", "type": "integer", "required": False, "default": 32},
        ],
        "output": "Completions",
    },
    {
        "name": "lean_declaration_file",
        "description": "Return the declaration file for a symbol.",
        "inputs": [
            {"name": "file_path", "type": "string", "required": True},
            {"name": "symbol", "type": "string", "required": True},
        ],
        "output": "Declaration",
    },
    {
        "name": "lean_multi_attempt",
        "description": "Apply multiple snippets at a line and compare results.",
        "inputs": [
            {"name": "file_path", "type": "string", "required": True},
            {"name": "line", "type": "integer", "required": True},
            {"name": "snippets", "type": "array", "items": {"type": "string"}, "required": True},
        ],
        "output": "MultiAttempt",
    },
    {
        "name": "lean_run_code",
        "description": "Run an isolated Lean snippet and report diagnostics.",
        "inputs": [{"name": "code", "type": "string", "required": True}],
        "output": "RunCode",
    },
    {
        "name": "lean_leansearch",
        "description": "Search Lean theorems via leansearch.net.",
        "inputs": [
            {"name": "query", "type": "string", "required": True},
            {"name": "num_results", "type": "integer", "required": False, "default": 5},
        ],
        "output": "SearchResults",
        "rate_limit": {"max": 3, "window_seconds": 30},
    },
    {
        "name": "lean_loogle",
        "description": "Search definitions and theorems via loogle.",
        "inputs": [
            {"name": "query", "type": "string", "required": True},
            {"name": "num_results", "type": "integer", "required": False, "default": 8},
        ],
        "output": "SearchResults",
        "rate_limit": {"max": 3, "window_seconds": 30},
    },
    {
        "name": "lean_state_search",
        "description": "Goal-directed search using premise-search.com.",
        "inputs": [
            {"name": "file_path", "type": "string", "required": True},
            {"name": "line", "type": "integer", "required": True},
            {"name": "column", "type": "integer", "required": True},
            {"name": "num_results", "type": "integer", "required": False, "default": 5},
        ],
        "output": "SearchResults",
        "rate_limit": {"max": 3, "window_seconds": 30},
    },
    {
        "name": "lean_hammer_premise",
        "description": "Fetch premises via the Lean hammer service.",
        "inputs": [
            {"name": "file_path", "type": "string", "required": True},
            {"name": "line", "type": "integer", "required": True},
            {"name": "column", "type": "integer", "required": True},
            {"name": "num_results", "type": "integer", "required": False, "default": 32},
        ],
        "output": "SearchResults",
        "rate_limit": {"max": 3, "window_seconds": 30},
    },
    {
        "name": "lean_tool_spec",
        "description": "Return this machine-readable tool specification.",
        "inputs": [],
        "output": "ToolSpecification",
    },
]

OUTPUT_SCHEMAS["ToolSpecification"] = {
    "description": "Specification metadata for all tools",
    "properties": {
        "version": {"type": "string"},
        "schema_version": {"type": "string"},
        "tools": {"type": "array"},
        "outputs": {"type": "object"},
    },
}


def build_tool_spec() -> Dict[str, Any]:
    return {
        "version": TOOL_SPEC_VERSION,
        "schema_version": SCHEMA_VERSION,
        "tools": TOOLS,
        "outputs": OUTPUT_SCHEMAS,
    }


__all__ = ["build_tool_spec", "TOOL_SPEC_VERSION"]
