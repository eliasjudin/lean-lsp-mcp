"""Typed response primitives for Lean LSP MCP."""

from __future__ import annotations

from typing import List, TypedDict

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
class Position(TypedDict):
    line: int
    column: int


class Range(TypedDict):
    start: Position
    end: Position


class DiagnosticEntry(TypedDict, total=False):
    message: str
    severity: int | None
    range: Range | None
    source: str
    code: str | int


class GoalPayload(TypedDict, total=False):
    rendered: str | None
    goals: List[str]
    user_state: str
    pp: str


class ErrorMeta(TypedDict, total=False):
    code: str
    retryable: bool


class RateLimitMeta(TypedDict):
    max_requests: int
    per_seconds: int


class ResponseMeta(TypedDict, total=False):
    error: ErrorMeta
    rate_limit: RateLimitMeta
    pagination: "PaginationMeta"


class PaginationMeta(TypedDict):
    start_line: int
    end_line: int
    total_lines: int
    has_more: bool
    next_start_line: int | None


__all__ = [
    "DiagnosticEntry",
    "ErrorMeta",
    "GoalPayload",
    "Position",
    "Range",
    "RateLimitMeta",
    "PaginationMeta",
    "ResponseMeta",
    "ERROR_BAD_REQUEST",
    "ERROR_CLIENT_NOT_READY",
    "ERROR_INVALID_PATH",
    "ERROR_IO_FAILURE",
    "ERROR_NO_GOAL",
    "ERROR_NOT_GOAL_POSITION",
    "ERROR_RATE_LIMIT",
    "ERROR_UNKNOWN",
]
