import os
import sys
import tempfile
import textwrap
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

from mcp.server.auth.provider import AccessToken, TokenVerifier

from lean_lsp_mcp.schema_types import (
    DiagnosticEntry,
    GoalPayload,
    PaginationMeta,
    Position,
    Range,
)


class StdoutToStderr:
    """Redirects stdout to stderr at the file descriptor level bc lake build logging"""

    def __init__(self):
        self.original_stdout_fd = None

    def __enter__(self):
        self.original_stdout_fd = os.dup(sys.stdout.fileno())
        stderr_fd = sys.stderr.fileno()
        os.dup2(stderr_fd, sys.stdout.fileno())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_stdout_fd is not None:
            os.dup2(self.original_stdout_fd, sys.stdout.fileno())
            os.close(self.original_stdout_fd)
            self.original_stdout_fd = None


class OutputCapture:
    """Capture any output to stdout and stderr at the file descriptor level."""

    def __init__(self):
        self.original_stdout_fd = None
        self.original_stderr_fd = None
        self.temp_file = None
        self.captured_output = ""

    def __enter__(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        self.original_stdout_fd = os.dup(sys.stdout.fileno())
        self.original_stderr_fd = os.dup(sys.stderr.fileno())
        os.dup2(self.temp_file.fileno(), sys.stdout.fileno())
        os.dup2(self.temp_file.fileno(), sys.stderr.fileno())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.original_stdout_fd, sys.stdout.fileno())
        os.dup2(self.original_stderr_fd, sys.stderr.fileno())
        os.close(self.original_stdout_fd)
        os.close(self.original_stderr_fd)

        self.temp_file.flush()
        self.temp_file.seek(0)
        self.captured_output = self.temp_file.read()
        self.temp_file.close()
        os.unlink(self.temp_file.name)

    def get_output(self):
        return self.captured_output


def _normalize_position(line: int, character: int) -> Position:
    """Return 1-indexed line/column payloads."""

    return {"line": line + 1, "column": character + 1}


def normalize_range(range_dict: Optional[Dict[str, Dict[str, int]]]) -> Optional[Range]:
    """Convert an LSP range (0-indexed) into 1-indexed coordinates."""

    if not range_dict:
        return None
    return {
        "start": _normalize_position(
            range_dict["start"]["line"], range_dict["start"]["character"]
        ),
        "end": _normalize_position(
            range_dict["end"]["line"], range_dict["end"]["character"]
        ),
    }


def compute_pagination(
    total_lines: int,
    start_line: Optional[int],
    line_count: Optional[int],
) -> tuple[int, int, PaginationMeta]:
    """Normalize pagination inputs and build response metadata."""

    if start_line is None or start_line < 1:
        start_line = 1
    if line_count is None or line_count < 1:
        line_count = total_lines - start_line + 1

    end_line = min(total_lines, start_line + line_count - 1)
    has_more = end_line < total_lines
    next_start = end_line + 1 if has_more else None
    meta: PaginationMeta = {
        "start_line": start_line,
        "end_line": end_line if total_lines else 0,
        "total_lines": total_lines,
        "has_more": has_more,
        "next_start_line": next_start,
    }
    return start_line, end_line, meta


def diagnostics_to_entries(
    diagnostics: List[Dict], select_line: int = -1
) -> List[DiagnosticEntry]:
    """Convert Lean diagnostics to structured entries."""

    entries: List[Dict[str, Any]] = []
    if select_line != -1:
        diagnostics = filter_diagnostics_by_position(diagnostics, select_line, None)

    for diag in diagnostics:
        primary_range = diag.get("fullRange", diag.get("range"))
        entry: DiagnosticEntry = {
            "message": diag.get("message", ""),
            "severity": diag.get("severity"),
            "range": normalize_range(primary_range),
        }
        if "source" in diag:
            entry["source"] = diag["source"]
        if "code" in diag:
            entry["code"] = diag["code"]
        entries.append(entry)
    return entries


_SEVERITY_LABELS = {
    1: "Error",
    2: "Warning",
    3: "Info",
    4: "Hint",
}


def _diagnostic_path(diag: Dict[str, Any]) -> Optional[str]:
    file_candidate = (
        diag.get("file")
        or diag.get("path")
        or diag.get("fileName")
        or diag.get("uri")
    )
    if isinstance(file_candidate, str) and file_candidate.startswith("file://"):
        parsed = urlparse(file_candidate)
        return unquote(parsed.path)
    if isinstance(file_candidate, str) and file_candidate.startswith("file:"):
        return unquote(file_candidate[5:])
    return file_candidate if isinstance(file_candidate, str) else None


def _format_range_positions(range_dict: Optional[Dict[str, Any]]) -> Optional[str]:
    if not range_dict:
        return None

    start = range_dict.get("start", {})
    end = range_dict.get("end", {})
    line_start = start.get("line")
    col_start = start.get("character")
    line_end = end.get("line")
    col_end = end.get("character")

    if line_start is None or col_start is None:
        return None

    start_label = f"{line_start + 1}:{col_start + 1}"
    if line_end is None or col_end is None:
        return start_label

    end_label = f"{line_end + 1}:{col_end + 1}"
    return f"{start_label}-{end_label}"


def _format_range_label(range_dict: Optional[Dict[str, Any]], diag: Dict[str, Any]) -> str:
    positions = _format_range_positions(range_dict)
    if not positions:
        return "No range"

    path = _diagnostic_path(diag)
    if path:
        return f"{path}:{positions}"
    return positions


def _format_related_information(related: List[Dict[str, Any]]) -> List[str]:
    formatted: List[str] = []
    for info in related:
        message = info.get("message", "").strip()
        location = info.get("location", {})
        range_dict = location.get("range")
        uri = location.get("uri")
        diag_stub = {"uri": uri} if uri else {}
        location_label = _format_range_label(range_dict, diag_stub)

        if message:
            formatted.append(f"{location_label}\n{message}")
        else:
            formatted.append(location_label)
    return formatted


def format_diagnostics(diagnostics: List[Dict], select_line: int = -1) -> List[str]:
    """Format diagnostics for legacy text responses."""

    if select_line != -1:
        diagnostics = filter_diagnostics_by_position(diagnostics, select_line, None)

    formatted_messages: List[str] = []
    for diag in diagnostics:
        severity_value = diag.get("severity")
        severity_label = _SEVERITY_LABELS.get(severity_value, str(severity_value))
        range_dict = diag.get("fullRange", diag.get("range"))
        location_label = _format_range_label(range_dict, diag)

        source = diag.get("source")
        code = diag.get("code")
        provenance_parts = [str(part) for part in (source, code) if part]
        provenance_suffix = f" ({'#'.join(provenance_parts)})" if provenance_parts else ""

        header = f"[{severity_label}] {location_label}{provenance_suffix}"

        message = diag.get("message", "")
        message_block = message.rstrip("\n")

        related = diag.get("relatedInformation") or []
        related_blocks = _format_related_information(related)
        related_text = [textwrap.indent(block, "  ") for block in related_blocks]

        block_lines = [header]
        if message_block:
            block_lines.append(message_block)
        if related_text:
            block_lines.extend(related_text)

        formatted_messages.append("\n".join(block_lines))

    return formatted_messages


def format_goal(goal, default_msg):
    if goal is None:
        return default_msg
    rendered = goal.get("rendered")
    return rendered.replace("```lean\n", "").replace("\n```", "") if rendered else None


def clean_rendered(goal: Optional[Dict[str, Any]]) -> Optional[str]:
    if goal is None:
        return None
    rendered = goal.get("rendered")
    if rendered is None:
        return None
    return rendered.replace("```lean\n", "").replace("\n```", "")


def goal_to_payload(goal: Optional[Dict[str, Any]]) -> Optional[GoalPayload]:
    if goal is None:
        return None
    payload: GoalPayload = {
        "rendered": clean_rendered(goal),
        "goals": goal.get("goals", []),
    }
    if "userState" in goal:
        payload["user_state"] = goal["userState"]
    if "pp" in goal:
        payload["pp"] = goal["pp"]
    return payload


def _utf16_index_to_py_index(text: str, utf16_index: int) -> int | None:
    """Convert an LSP UTF-16 column index into a Python string index."""
    if utf16_index < 0:
        return None

    units = 0
    for idx, ch in enumerate(text):
        code_point = ord(ch)
        next_units = units + (2 if code_point > 0xFFFF else 1)

        if utf16_index < next_units:
            return idx
        if utf16_index == next_units:
            return idx + 1

        units = next_units
    if units >= utf16_index:
        return len(text)
    return None


def extract_range(content: str, range: dict) -> str:
    """Extract the text from the content based on the range.

    Args:
        content (str): The content to extract from.
        range (dict): The range to extract.

    Returns:
        str: The extracted range text.
    """
    start_line = range["start"]["line"]
    start_char = range["start"]["character"]
    end_line = range["end"]["line"]
    end_char = range["end"]["character"]

    lines = content.splitlines(keepends=True)
    if not lines:
        lines = [""]

    line_offsets: List[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)
    total_length = len(content)

    def position_to_offset(line: int, character: int) -> int | None:
        if line == len(lines) and character == 0:
            return total_length
        if line < 0 or line >= len(lines):
            return None
        py_index = _utf16_index_to_py_index(lines[line], character)
        if py_index is None:
            return None
        if py_index > len(lines[line]):
            return None
        return line_offsets[line] + py_index

    start_offset = position_to_offset(start_line, start_char)
    end_offset = position_to_offset(end_line, end_char)

    if start_offset is None or end_offset is None or start_offset > end_offset:
        return "Range out of bounds"

    return content[start_offset:end_offset]


def find_start_position(content: str, query: str) -> dict | None:
    """Find the position of the query in the content.

    Args:
        content (str): The content to search in.
        query (str): The query to find.

    Returns:
        dict | None: The position of the query in the content. {"line": int, "column": int}
    """
    lines = content.splitlines()
    for line_number, line in enumerate(lines):
        char_index = line.find(query)
        if char_index != -1:
            return {"line": line_number, "column": char_index}
    return None


def format_line(
    file_content: str,
    line_number: int,
    column: Optional[int] = None,
    cursor_tag: Optional[str] = "<cursor>",
) -> str:
    """Show a line and cursor position in a file.

    Args:
        file_content (str): The content of the file.
        line_number (int): The line number (1-indexed).
        column (Optional[int]): The column number (1-indexed). If None, no cursor position is shown.
        cursor_tag (Optional[str]): The tag to use for the cursor position. Defaults to "<cursor>".
    Returns:
        str: The formatted position.
    """
    lines = file_content.splitlines()
    line_number -= 1
    if line_number < 0 or line_number >= len(lines):
        return "Line number out of range"
    line = lines[line_number]
    if column is None:
        return line
    column -= 1
    # Allow placing the cursor at end-of-line (column == len(line))
    if column < 0 or column > len(line):
        return "Invalid column number"
    return f"{line[:column]}{cursor_tag}{line[column:]}"


def filter_diagnostics_by_position(
    diagnostics: List[Dict], line: int, column: Optional[int]
) -> List[Dict]:
    """Find diagnostics at a specific position.

    Args:
        diagnostics (List[Dict]): List of diagnostics.
        line (int): The line number (0-indexed).
        column (Optional[int]): The column number (0-indexed).

    Returns:
        List[Dict]: List of diagnostics at the specified position.
    """
    if column is None:
        return [
            d
            for d in diagnostics
            if d["range"]["start"]["line"] <= line <= d["range"]["end"]["line"]
        ]

    return [
        d
        for d in diagnostics
        if d["range"]["start"]["line"] <= line <= d["range"]["end"]["line"]
        and d["range"]["start"]["character"] <= column < d["range"]["end"]["character"]
    ]


class OptionalTokenVerifier(TokenVerifier):
    def __init__(self, expected_token: str):
        self.expected_token = expected_token

    async def verify_token(self, token: str) -> AccessToken | None:
        if token == self.expected_token:
            return AccessToken(token=token, client_id="lean-lsp-mcp", scopes=["user"])
        return None
