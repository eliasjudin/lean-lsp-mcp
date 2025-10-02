"""Typed response primitives for Lean LSP MCP."""

from __future__ import annotations

from typing import Dict, List, TypedDict

# ----- Error codes ---------------------------------------------------------
ERROR_BAD_REQUEST = "bad_request"
ERROR_INVALID_PATH = "invalid_path"
ERROR_NO_GOAL = "no_goal"
ERROR_NOT_GOAL_POSITION = "not_goal_position"
ERROR_RATE_LIMIT = "rate_limited"
ERROR_CLIENT_NOT_READY = "client_not_ready"
ERROR_IO_FAILURE = "io_failure"
ERROR_UNKNOWN = "unknown"


# ----- Shared structures ---------------------------------------------------
class LSPPosition(TypedDict):
    line: int
    character: int


class LSPRange(TypedDict):
    start: LSPPosition
    end: LSPPosition


class FileIdentity(TypedDict):
    uri: str
    relative_path: str


class DiagnosticEntry(TypedDict, total=False):
    message: str
    severity: str | None
    severityCode: int | None
    range: LSPRange | None
    source: str
    code: str | int
    tags: List[str]
    relatedInformation: List[Dict]


class DiagnosticsSummary(TypedDict):
    count: int
    bySeverity: Dict[str, int]
    has_errors: bool


class DiagnosticsPayload(TypedDict):
    file: FileIdentity
    diagnostics: List[DiagnosticEntry]
    summary: DiagnosticsSummary


class GoalPayload(TypedDict, total=False):
    rendered: str | None
    goals: List[str]
    user_state: str
    pp: str


class PaginationMeta(TypedDict, total=False):
    start_line: int
    end_line: int
    total_lines: int
    has_more: bool
    next_start_line: int | None
    nextCursor: str


__all__ = [
    "DiagnosticEntry",
    "DiagnosticsPayload",
    "DiagnosticsSummary",
    "FileIdentity",
    "GoalPayload",
    "LSPPosition",
    "LSPRange",
    "PaginationMeta",
    "ERROR_BAD_REQUEST",
    "ERROR_CLIENT_NOT_READY",
    "ERROR_INVALID_PATH",
    "ERROR_IO_FAILURE",
    "ERROR_NO_GOAL",
    "ERROR_NOT_GOAL_POSITION",
    "ERROR_RATE_LIMIT",
    "ERROR_UNKNOWN",
]
