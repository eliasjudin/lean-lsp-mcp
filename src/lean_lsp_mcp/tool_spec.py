"""Lightweight machine-readable tool specification."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Type

from pydantic import BaseModel

from lean_lsp_mcp.tools import TOOL_ANNOTATIONS, TOOL_DEFINITIONS

TOOL_SPEC_VERSION = "2025-10-20.1"

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

    ref_value = prop.get("$ref")
    if isinstance(ref_value, str):
        return ref_value.split("/")[-1]
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

        # Surface alternate accepted names (validation aliases)
        alt_names: List[str] = []
        try:
            va = getattr(field, "validation_alias", None)
            # Pydantic v2 AliasChoices exposes `.choices`
            choices = getattr(va, "choices", None)
            if isinstance(choices, (list, tuple)):
                for choice in choices:
                    # Choice can be a string or an AliasPath-like object
                    if isinstance(choice, str):
                        alt_names.append(choice)
                    else:
                        # Try to extract the first component name
                        path = getattr(choice, "path", None) or getattr(choice, "components", None)
                        if isinstance(path, (list, tuple)) and path and isinstance(path[0], str):
                            alt_names.append(path[0])
            elif isinstance(va, str):
                alt_names.append(va)
        except Exception:
            pass

        # Also include the declared field alias if different from canonical
        if field.alias and field.alias not in (alias, field_name):
            alt_names.append(field.alias)

        # Remove duplicates and the canonical name
        alt_names = [n for n in dict.fromkeys(alt_names) if n and n != alias]
        if alt_names:
            entry["aliases"] = alt_names

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
        entry["annotations"] = annotations.model_dump(exclude_none=True)
    # Surface example payloads to guide UIs; prefer canonical names like `column`.
    examples = meta.get("examples")
    if isinstance(examples, list) and examples:
        entry["examples"] = examples
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


@lru_cache(maxsize=1)
def _cached_tool_spec() -> Dict[str, Any]:
    tools = [_build_tool_entry(name, meta) for name, meta in TOOL_DEFINITIONS]
    return {
        "version": TOOL_SPEC_VERSION,
        "tools": tools,
        "responses": RESPONSE_KIND_SUMMARY,
    }


def build_tool_spec() -> Dict[str, Any]:
    return deepcopy(_cached_tool_spec())


__all__ = ["build_tool_spec", "TOOL_SPEC_VERSION", "TOOL_ANNOTATIONS"]


if __name__ == "__main__":
    import json
    import sys

    json.dump(build_tool_spec(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
