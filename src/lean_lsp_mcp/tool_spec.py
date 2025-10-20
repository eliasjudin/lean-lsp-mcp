"""Lightweight machine-readable tool specification."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Type

from pydantic import BaseModel

from lean_lsp_mcp.tool_inputs import (
    LeanBuildInput,
    LeanCompletionsInput,
    LeanDeclarationFileInput,
    LeanDiagnosticMessagesInput,
    LeanFileContentsInput,
    LeanGoalInput,
    LeanHammerPremiseInput,
    LeanHoverInput,
    LeanMultiAttemptInput,
    LeanRunCodeInput,
    LeanSearchInput,
    LeanStateSearchInput,
    LeanTermGoalInput,
    LoogleSearchInput,
    LeanToolSpecInput,
)

TOOL_SPEC_VERSION = "2025-10-20.1"


TOOL_ANNOTATIONS: Dict[str, Dict[str, Any]] = {
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


def _primary_type(prop: Dict[str, Any]) -> str:
    """Collapse common JSON Schema representations into a readable type label."""

    type_value = prop.get("type")
    if isinstance(type_value, list):
        return " | ".join(type_value)
    if isinstance(type_value, str):
        return type_value

    for key in ("const", "enum"):
        if key in prop:
            return key

    any_of = prop.get("anyOf") or prop.get("oneOf") or prop.get("allOf")
    if isinstance(any_of, list):
        collected = []
        for item in any_of:
            if isinstance(item, dict):
                inner_type = item.get("type")
                if inner_type:
                    collected.append(inner_type)
        if collected:
            return " | ".join(dict.fromkeys(collected))  # Preserve order, remove dupes.

    if "$ref" in prop:
        return prop["$ref"].split("/")[-1]
    return "object"


def _model_to_inputs(model: Type[BaseModel]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return simplified field metadata and the full JSON schema for a model."""

    schema = model.model_json_schema()
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    items: List[Dict[str, Any]] = []

    # Iterate over declared fields to preserve definition order.
    for field_name, field in model.model_fields.items():
        alias = field.serialization_alias or field.alias or field_name
        prop_schema = properties.get(alias)
        if not isinstance(prop_schema, dict):
            continue

        entry: Dict[str, Any] = {
            "name": alias,
            "required": alias in required,
            "type": _primary_type(prop_schema),
        }
        if alias != field_name:
            entry["field"] = field_name

        description = prop_schema.get("description")
        if description:
            entry["description"] = description

        if "default" in prop_schema:
            entry["default"] = prop_schema["default"]

        for constraint_key in (
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "minLength",
            "maxLength",
            "pattern",
            "minItems",
            "maxItems",
        ):
            if constraint_key in prop_schema:
                entry[constraint_key] = prop_schema[constraint_key]

        if "enum" in prop_schema:
            entry["enum"] = prop_schema["enum"]
        if "items" in prop_schema:
            entry["items"] = prop_schema["items"]
        if "anyOf" in prop_schema:
            entry["anyOf"] = prop_schema["anyOf"]

        items.append(entry)

    return items, schema


TOOL_DEFINITIONS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "lean_build",
        {
            "description": "Run `lake build` (optionally `lake clean`) to refresh the Lean project and restart the cached Lean LSP client.",
            "response": "LeanBuildResult",
            "model": LeanBuildInput,
            "annotations": TOOL_ANNOTATIONS["lean_build"],
        },
    ),
    (
        "lean_file_contents",
        {
            "description": "Read Lean source text with optional 1-indexed annotations; `structured.lines` preserves numbering for downstream agents.",
            "response": "FileContents",
            "model": LeanFileContentsInput,
            "annotations": TOOL_ANNOTATIONS["lean_file_contents"],
        },
    ),
    (
        "lean_diagnostic_messages",
        {
            "description": "List Lean diagnostics for a file window; `structured.summary.count` reports totals and `structured.diags` encode zero-based ranges for tooling.",
            "response": "Diagnostics",
            "model": LeanDiagnosticMessagesInput,
            "annotations": TOOL_ANNOTATIONS["lean_diagnostic_messages"],
        },
    ),
    (
        "lean_goal",
        {
            "description": "Inspect Lean proof goals at a 1-indexed file position; omitting the column probes line boundaries and returns compact `goals` entries with first-line previews.",
            "response": "Goal",
            "model": LeanGoalInput,
            "annotations": TOOL_ANNOTATIONS["lean_goal"],
        },
    ),
    (
        "lean_term_goal",
        {
            "description": "Inspect the expected Lean term type at a 1-indexed location and expose the cleaned `rendered` snippet alongside normalized coordinates.",
            "response": "Goal",
            "model": LeanTermGoalInput,
            "annotations": TOOL_ANNOTATIONS["lean_term_goal"],
        },
    ),
    (
        "lean_hover_info",
        {
            "description": "Retrieve Lean hover documentation for the symbol under the cursor together with `structured.infoSnippet` and nearby diagnostics.",
            "response": "Hover",
            "model": LeanHoverInput,
            "annotations": TOOL_ANNOTATIONS["lean_hover_info"],
        },
    ),
    (
        "lean_completions",
        {
            "description": "Request Lean completion suggestions at a file position.",
            "response": "Completions",
            "model": LeanCompletionsInput,
            "annotations": TOOL_ANNOTATIONS["lean_completions"],
        },
    ),
    (
        "lean_declaration_file",
        {
            "description": "Open the Lean source file that defines a given declaration and surface declaration path metadata for navigation.",
            "response": "Declaration",
            "model": LeanDeclarationFileInput,
            "annotations": TOOL_ANNOTATIONS["lean_declaration_file"],
        },
    ),
    (
        "lean_multi_attempt",
        {
            "description": "Evaluate multiple Lean snippets at a line and compare diagnostics/goals.",
            "response": "MultiAttempt",
            "model": LeanMultiAttemptInput,
            "annotations": TOOL_ANNOTATIONS["lean_multi_attempt"],
        },
    ),
    (
        "lean_run_code",
        {
            "description": "Execute an isolated Lean snippet with standalone diagnostics.",
            "response": "RunCode",
            "model": LeanRunCodeInput,
            "annotations": TOOL_ANNOTATIONS["lean_run_code"],
        },
    ),
    (
        "lean_tool_spec",
        {
            "description": "Publish structured metadata for all Lean MCP tools, including inputs, annotations, and rate limits.",
            "response": "ToolSpecSummary",
            "model": LeanToolSpecInput,
            "annotations": TOOL_ANNOTATIONS["lean_tool_spec"],
        },
    ),
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
            "rate_limit": {"category": "lean_state_search", "max": 3, "window_seconds": 30},
            "annotations": TOOL_ANNOTATIONS["lean_state_search"],
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
        },
    ),
]


def _build_tool_entry(name: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    model_cls: Type[BaseModel] = meta["model"]
    inputs, schema = _model_to_inputs(model_cls)

    entry: Dict[str, Any] = {
        "name": name,
        "description": meta["description"],
        "response": meta["response"],
        "inputModel": model_cls.__name__,
        "inputs": inputs,
        "inputSchema": schema,
    }
    if "rate_limit" in meta:
        entry["rate_limit"] = meta["rate_limit"]
    annotations = meta.get("annotations")
    if annotations:
        entry["annotations"] = dict(annotations)
    return entry


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
    "ToolSpecSummary": {"description": "Lean MCP tool specification payload."},
}


def build_tool_spec() -> Dict[str, Any]:
    tools = [_build_tool_entry(name, meta) for name, meta in TOOL_DEFINITIONS]
    return {
        "version": TOOL_SPEC_VERSION,
        "tools": tools,
        "responses": RESPONSE_KIND_SUMMARY,
    }


__all__ = ["build_tool_spec", "TOOL_SPEC_VERSION", "TOOL_ANNOTATIONS"]


if __name__ == "__main__":
    import json
    import sys

    json.dump(build_tool_spec(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
