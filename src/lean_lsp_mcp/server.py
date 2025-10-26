"""Compatibility layer exporting server components.

This module preserves the historical public API where tool callables and
input dataclasses were exposed from ``lean_lsp_mcp.server``.  The concrete
implementations now live under ``lean_lsp_mcp.server_components``.
"""

from __future__ import annotations

import urllib
from importlib import import_module
from typing import Any, ContextManager, Dict, List

from lean_lsp_mcp.client_utils import setup_client_for_file
from lean_lsp_mcp.leanclient_provider import LeanclientNotInstalledError
from lean_lsp_mcp.schema_types import (
    ERROR_BAD_REQUEST,
    ERROR_CLIENT_NOT_READY,
    ERROR_INVALID_PATH,
    ERROR_IO_FAILURE,
    ERROR_NO_GOAL,
    ERROR_NOT_GOAL_POSITION,
    ERROR_RATE_LIMIT,
    ERROR_UNKNOWN,
)
from lean_lsp_mcp.server_components import common as _common
from lean_lsp_mcp.server_components import mcp as mcp
from lean_lsp_mcp.server_components.common import (
    LeanFileSession,
    ToolError,
    _compact_pos,
    _derive_error_hints,
    _identity_for_rel_path,
    _json_item,
    _resource_item,
    _sanitize_path_label,
    _set_response_format_hint,
    _text_item,
    error_result,
    rate_limited,
    register_server_module,
    success_result,
)
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
    LeanToolSpecInput,
    LoogleSearchInput,
)

# Legacy module aliases for backwards compatibility
common = _common

time = _common.time
sanitize_path_label = _sanitize_path_label
register_server_module(globals())


def _ensure_file_tools() -> Any:
    module = globals().get("_file_tools")
    if module is None:
        module = import_module("lean_lsp_mcp.server_components.file_tools")
        globals()["_file_tools"] = module
        globals()["file_tools"] = module
        globals()["lean_file_contents"] = module.lean_file_contents
        globals()["lean_diagnostic_messages"] = module.lean_diagnostic_messages
        globals()["file_contents"] = module.lean_file_contents
    return module


def _ensure_goal_tools() -> Any:
    module = globals().get("_goal_tools")
    if module is None:
        module = import_module("lean_lsp_mcp.server_components.goal_tools")
        globals()["_goal_tools"] = module
        globals()["goal_tools"] = module
        globals()["lean_goal"] = module.goal
        globals()["lean_term_goal"] = module.term_goal
    return module


def _ensure_hover_completion_tools() -> Any:
    module = globals().get("_hover_completion_tools")
    if module is None:
        module = import_module("lean_lsp_mcp.server_components.hover_completion_tools")
        globals()["_hover_completion_tools"] = module
        globals()["hover_completion_tools"] = module
        globals()["lean_hover"] = module.hover
        globals()["lean_completions"] = module.completions
    return module


def _ensure_declaration_tools() -> Any:
    module = globals().get("_declaration_tools")
    if module is None:
        module = import_module("lean_lsp_mcp.server_components.declaration_tools")
        globals()["_declaration_tools"] = module
        globals()["declaration_tools"] = module
        globals()["lean_declaration_file"] = module.declaration_file
    return module


def _ensure_snippet_tools() -> Any:
    module = globals().get("_snippet_tools")
    if module is None:
        module = import_module("lean_lsp_mcp.server_components.snippet_tools")
        globals()["_snippet_tools"] = module
        globals()["snippet_tools"] = module
        globals()["lean_multi_attempt"] = module.multi_attempt
        globals()["lean_run_code"] = module.run_code
    return module


def _ensure_project_tools() -> Any:
    module = globals().get("_project_tools")
    if module is None:
        module = import_module("lean_lsp_mcp.server_components.project_tools")
        globals()["_project_tools"] = module
        globals()["project_tools"] = module
        globals()["lean_build"] = module.lean_build
    return module


def _ensure_spec_tools() -> Any:
    module = globals().get("_spec_tools")
    if module is None:
        module = import_module("lean_lsp_mcp.server_components.spec_tools")
        globals()["_spec_tools"] = module
        globals()["spec_tools"] = module
        globals()["build_tool_spec"] = module.build_tool_spec
        globals()["lean_tool_spec"] = module.tool_spec
    return module


def _ensure_search_tools() -> Any:
    module = globals().get("_search_tools")
    if module is None:
        module = import_module("lean_lsp_mcp.server_components.search_tools")
        globals()["_search_tools"] = module
        globals()["search_tools"] = module
        globals()["leansearch"] = module.leansearch
        globals()["loogle"] = module.loogle
        globals()["lean_state_search"] = module.state_search
        globals()["hammer_premise"] = module.hammer_premise
    return module


def _ensure_context() -> Any:
    module = globals().get("_context")
    if module is None:
        module = import_module("lean_lsp_mcp.server_components.context")
        globals()["_context"] = module
        globals()["client_session"] = module.client_session
    return module


_LAZY_ATTRS = {
    "_file_tools": _ensure_file_tools,
    "file_tools": _ensure_file_tools,
    "lean_file_contents": _ensure_file_tools,
    "lean_diagnostic_messages": _ensure_file_tools,
    "file_contents": _ensure_file_tools,
    "_goal_tools": _ensure_goal_tools,
    "goal_tools": _ensure_goal_tools,
    "lean_goal": _ensure_goal_tools,
    "lean_term_goal": _ensure_goal_tools,
    "_hover_completion_tools": _ensure_hover_completion_tools,
    "hover_completion_tools": _ensure_hover_completion_tools,
    "lean_hover": _ensure_hover_completion_tools,
    "lean_completions": _ensure_hover_completion_tools,
    "_declaration_tools": _ensure_declaration_tools,
    "declaration_tools": _ensure_declaration_tools,
    "lean_declaration_file": _ensure_declaration_tools,
    "_snippet_tools": _ensure_snippet_tools,
    "snippet_tools": _ensure_snippet_tools,
    "lean_multi_attempt": _ensure_snippet_tools,
    "lean_run_code": _ensure_snippet_tools,
    "_project_tools": _ensure_project_tools,
    "project_tools": _ensure_project_tools,
    "lean_build": _ensure_project_tools,
    "_spec_tools": _ensure_spec_tools,
    "spec_tools": _ensure_spec_tools,
    "build_tool_spec": _ensure_spec_tools,
    "lean_tool_spec": _ensure_spec_tools,
    "_search_tools": _ensure_search_tools,
    "search_tools": _ensure_search_tools,
    "leansearch": _ensure_search_tools,
    "loogle": _ensure_search_tools,
    "lean_state_search": _ensure_search_tools,
    "hammer_premise": _ensure_search_tools,
    "_context": _ensure_context,
    "client_session": _ensure_context,
}



def __getattr__(name: str) -> Any:
    loader = _LAZY_ATTRS.get(name)
    if loader is not None:
        loader()
        return globals()[name]
    raise AttributeError(f"module 'lean_lsp_mcp.server' has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__) | set(_LAZY_ATTRS))


def lean_snippet(ctx: Any, params: Any) -> Any:
    module = _ensure_snippet_tools()
    return module.run_code(ctx, params)


def lean_snippet_apply(ctx: Any, params: Any) -> Any:
    module = _ensure_snippet_tools()
    return module.multi_attempt(ctx, params)




# Ensure monkeypatches on this module are always picked up by common.open_file_session
def open_file_session(
    ctx: Any,
    file_path: str,
    *,
    started: float,
    response_format: str | None = None,
    line: int | None = None,
    column: int | None = None,
    category: str | None = None,
    invalid_details: Dict[str, Any] | None = None,
    client_details: Dict[str, Any] | None = None,
    invalid_message: str = "Invalid Lean file path: Unable to start LSP server or load file",
    client_message: str = (
        "Lean client is not available. Run another tool to initialize the project first."
    ),
    ) -> ContextManager[LeanFileSession]:
    # Refresh registration in case multiple server modules were loaded in tests
    register_server_module(globals())
    return _common.open_file_session(
        ctx,
        file_path,
        started=started,
        response_format=response_format,
        line=line,
        column=column,
        category=category,
        invalid_details=invalid_details,
        client_details=client_details,
        invalid_message=invalid_message,
        client_message=client_message,
    )
# ruff: noqa: F822
__all__ = [
    "mcp",
    "ToolError",
    "LeanFileSession",
    "_set_response_format_hint",
    "_text_item",
    "_resource_item",
    "_json_item",
    "success_result",
    "_derive_error_hints",
    "error_result",
    "_identity_for_rel_path",
    "_compact_pos",
    "open_file_session",
    "rate_limited",
    "client_session",
    "time",
    "urllib",
    "lean_file_contents",
    "lean_diagnostic_messages",
    "file_contents",
    "lean_goal",
    "lean_term_goal",
    "lean_multi_attempt",
    "lean_hover",
    "lean_completions",
    "lean_declaration_file",
    "lean_run_code",
    "lean_snippet",
    "lean_snippet_apply",
    "lean_build",
    "lean_tool_spec",
    "build_tool_spec",
    "leansearch",
    "loogle",
    "lean_state_search",
    "hammer_premise",
    "LeanBuildInput",
    "LeanCompletionsInput",
    "LeanDeclarationFileInput",
    "LeanDiagnosticMessagesInput",
    "LeanFileContentsInput",
    "LeanGoalInput",
    "LeanHammerPremiseInput",
    "LeanHoverInput",
    "LeanMultiAttemptInput",
    "LeanRunCodeInput",
    "LeanSearchInput",
    "LeanStateSearchInput",
    "LeanTermGoalInput",
    "LeanToolSpecInput",
    "LoogleSearchInput",
    "setup_client_for_file",
    "LeanclientNotInstalledError",
    "common",
    "file_tools",
    "goal_tools",
    "hover_completion_tools",
    "declaration_tools",
    "snippet_tools",
    "project_tools",
    "spec_tools",
    "search_tools",
    "sanitize_path_label",
    "ERROR_BAD_REQUEST",
    "ERROR_CLIENT_NOT_READY",
    "ERROR_INVALID_PATH",
    "ERROR_IO_FAILURE",
    "ERROR_NO_GOAL",
    "ERROR_NOT_GOAL_POSITION",
    "ERROR_RATE_LIMIT",
    "ERROR_UNKNOWN",
]


if __name__ == "__main__":
    mcp.run()
