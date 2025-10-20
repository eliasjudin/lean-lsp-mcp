import base64
import contextlib
import io
import os
import sys
import tempfile
import textwrap
from collections import Counter
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import unquote, urlparse

from mcp.server.auth.provider import AccessToken, TokenVerifier

from lean_lsp_mcp.schema_types import (
    DiagnosticEntry,
    DiagnosticsSummary,
    FileIdentity,
    GoalPayload,
    LSPRange,
    PaginationMeta,
)


class StdoutToStderr:
    """Redirects stdout to stderr at the file descriptor level bc lake build logging"""

    def __init__(self):
        self.original_stdout_fd = None
        self._fallback = False
        self._original_stdout = None

    def __enter__(self):
        stdout = sys.stdout
        try:
            stdout_fd = stdout.fileno()
            stderr_fd = sys.stderr.fileno()
        except (AttributeError, io.UnsupportedOperation):
            stdout_fd = None

        if stdout_fd is None:
            self._fallback = True
            self._original_stdout = stdout

            class _StderrWriter:
                def __init__(self, original):
                    self._original = original

                def write(self, data):
                    sys.stderr.write(data)

                def flush(self):
                    sys.stderr.flush()

                def __getattr__(self, name):
                    try:
                        return getattr(self._original, name)
                    except AttributeError:
                        return getattr(sys.stderr, name)

            sys.stdout = _StderrWriter(stdout)
        else:
            self.original_stdout_fd = os.dup(stdout_fd)
            os.dup2(stderr_fd, stdout_fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fallback:
            if self._original_stdout is not None:
                sys.stdout = self._original_stdout
                self._original_stdout = None
            self._fallback = False
        else:
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
        self._fallback = False
        self._stdout_redirect = None
        self._stderr_redirect = None
        self._fallback_stream: io.StringIO | None = None

    def __enter__(self):
        try:
            stdout_fd = sys.stdout.fileno()
            stderr_fd = sys.stderr.fileno()
        except (AttributeError, io.UnsupportedOperation):
            stdout_fd = None

        if stdout_fd is None:
            self._fallback = True
            self._fallback_stream = io.StringIO()
            self._stdout_redirect = contextlib.redirect_stdout(self._fallback_stream)
            self._stderr_redirect = contextlib.redirect_stderr(self._fallback_stream)
            self._stdout_redirect.__enter__()
            self._stderr_redirect.__enter__()
        else:
            self.temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)
            self.original_stdout_fd = os.dup(stdout_fd)
            self.original_stderr_fd = os.dup(stderr_fd)
            os.dup2(self.temp_file.fileno(), stdout_fd)
            os.dup2(self.temp_file.fileno(), stderr_fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fallback:
            try:
                if self._stderr_redirect is not None:
                    self._stderr_redirect.__exit__(exc_type, exc_val, exc_tb)
                if self._stdout_redirect is not None:
                    self._stdout_redirect.__exit__(exc_type, exc_val, exc_tb)
            finally:
                self._stderr_redirect = None
                self._stdout_redirect = None
            if self._fallback_stream is not None:
                self.captured_output = self._fallback_stream.getvalue()
                self._fallback_stream.close()
                self._fallback_stream = None
            self._fallback = False
            return

        try:
            if self.original_stdout_fd is not None:
                os.dup2(self.original_stdout_fd, sys.stdout.fileno())
            if self.original_stderr_fd is not None:
                os.dup2(self.original_stderr_fd, sys.stderr.fileno())
        finally:
            if self.original_stdout_fd is not None:
                os.close(self.original_stdout_fd)
                self.original_stdout_fd = None
            if self.original_stderr_fd is not None:
                os.close(self.original_stderr_fd)
                self.original_stderr_fd = None

        if self.temp_file is not None:
            try:
                self.temp_file.flush()
                self.temp_file.seek(0)
                self.captured_output = self.temp_file.read()
            finally:
                name = self.temp_file.name
                self.temp_file.close()
                try:
                    os.unlink(name)
                except FileNotFoundError:  # pragma: no cover - best effort cleanup
                    pass
                self.temp_file = None

    def get_output(self):
        return self.captured_output


def normalize_range(range_dict: Optional[Dict[str, Dict[str, int]]]) -> Optional[LSPRange]:
    """Return a shallow copy of an LSP range using 1-based indexing."""

    if not range_dict:
        return None

    try:
        start = range_dict["start"]
        end = range_dict["end"]
        start_line = int(start["line"])
        start_char = int(start.get("character", 0))
        end_line = int(end["line"])
        end_char = int(end.get("character", 0))
        return {
            "start": {
                "line": start_line + 1,
                "character": start_char + 1,
            },
            "end": {
                "line": end_line + 1,
                "character": end_char + 1,
            },
        }
    except (KeyError, TypeError, ValueError):  # pragma: no cover - defensive guard
        return None


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
    meta: PaginationMeta = {
        "start_line": start_line,
        "end_line": end_line if total_lines else 0,
        "total_lines": total_lines,
        "has_more": has_more,
    }
    if has_more:
        next_line = end_line + 1
        meta["next_start_line"] = next_line
        try:
            cursor_bytes = str(next_line).encode("ascii")
            meta["nextCursor"] = base64.urlsafe_b64encode(cursor_bytes).decode("ascii")
        except Exception:  # pragma: no cover - defensive guard
            pass
    return start_line, end_line, meta


def _filter_diagnostics(
    diagnostics: Iterable[Dict[str, Any]],
    line: Optional[int],
    column: Optional[int],
) -> List[Dict[str, Any]]:
    """Return diagnostics that intersect the requested (0-indexed) position."""

    if line is None:
        return list(diagnostics)

    matches: List[Dict[str, Any]] = []
    for diagnostic in diagnostics:
        diagnostic_range = diagnostic.get("range")
        if not diagnostic_range:
            continue

        start = diagnostic_range.get("start", {})
        end = diagnostic_range.get("end", {})
        start_line = start.get("line")
        end_line = end.get("line")

        if start_line is None or end_line is None:
            continue
        if not (start_line <= line <= end_line):
            continue

        if column is None:
            matches.append(diagnostic)
            continue

        start_char = start.get("character", 0)
        end_char = end.get("character", column + 1)

        if line == start_line and column < start_char:
            continue
        if line == end_line and column >= end_char:
            continue

        matches.append(diagnostic)

    return matches


def diagnostics_to_entries(
    diagnostics: List[Dict], select_line: int = -1, column: Optional[int] = None
) -> List[DiagnosticEntry]:
    """Convert Lean diagnostics to structured entries."""

    if select_line != -1:
        diagnostics = _filter_diagnostics(diagnostics, select_line, column)

    entries: List[DiagnosticEntry] = []
    for diag in diagnostics:
        primary_range = diag.get("fullRange", diag.get("range"))
        severity_label, severity_code = _map_severity(diag.get("severity"))
        entry: DiagnosticEntry = {
            "message": diag.get("message", ""),
            "severity": severity_label,
            "severityCode": severity_code,
            "range": normalize_range(primary_range),
        }
        if "source" in diag:
            entry["source"] = diag["source"]
        if "code" in diag:
            entry["code"] = diag["code"]
        if "tags" in diag and isinstance(diag["tags"], list):
            entry["tags"] = list(diag["tags"])
        related = diag.get("relatedInformation")
        if isinstance(related, list):
            entry["relatedInformation"] = related
        entries.append(entry)
    return entries


def summarize_diagnostics(entries: Iterable[DiagnosticEntry]) -> DiagnosticsSummary:
    counts = Counter()
    for entry in entries:
        label = entry.get("severity") or "unknown"
        counts[label] += 1
    summary_counts: Dict[str, int] = dict(counts)
    total = sum(summary_counts.values())
    return {
        "count": total,
        "bySeverity": summary_counts,
        "has_errors": summary_counts.get("error", 0) > 0,
    }


_SEVERITY_LABELS = {
    1: "error",
    2: "warning",
    3: "info",
    4: "hint",
}


def _sanitize_relative_path(path: str) -> str:
    if not path:
        return path
    # Normalize Windows-style separators first so backslashes are treated as
    # path separators even on POSIX. Then construct a POSIX path.
    # Example: "Foo\\Bar/Proof.lean" -> "Foo/Bar/Proof.lean"
    normalized = path.replace("\\", "/")
    return PurePosixPath(normalized).as_posix()


def _absolute_path_to_uri(path: str) -> str:
    expanded = os.path.expanduser(path)
    path_obj = Path(expanded)
    if not path_obj.is_absolute():
        path_obj = (Path.cwd() / path_obj).resolve()
    else:
        path_obj = path_obj.resolve()
    return path_obj.as_uri()


def uri_to_absolute_path(uri: str | None) -> str | None:
    """Best-effort conversion from a file URI to an absolute filesystem path."""

    if not uri:
        return None

    try:
        parsed = urlparse(uri)
    except Exception:
        return None

    scheme = parsed.scheme
    if scheme and scheme != "file":
        return None

    netloc = parsed.netloc or ""
    path = parsed.path or ""

    if netloc and netloc not in ("", "localhost"):
        if os.name == "nt":
            path = f"//{netloc}{path}"
        else:
            path = f"//{netloc}{path}"

    path = unquote(path)

    if os.name == "nt":
        # Windows URIs often start with "/C:/..." – strip the leading slash.
        if path.startswith("/") and len(path) >= 3 and path[2] == ":":
            path = path[1:]
        path = path.replace("/", "\\")

    if not path:
        return None

    try:
        return os.path.abspath(os.path.expanduser(path))
    except (TypeError, ValueError):
        return None


def file_identity(relative_path: str, absolute_path: Optional[str] = None) -> FileIdentity:
    sanitized = _sanitize_relative_path(relative_path)
    uri = ""
    try:
        if absolute_path:
            uri = _absolute_path_to_uri(absolute_path)
        else:
            expanded = os.path.expanduser(sanitized)
            if expanded and os.path.isabs(expanded):
                uri = _absolute_path_to_uri(expanded)
    except ValueError:
        uri = ""
    return {
        "uri": uri,
        "relative_path": sanitized,
    }


def _map_severity(value: Any) -> tuple[str | None, int | None]:
    try:
        severity_int = int(value)
    except (TypeError, ValueError):
        return None, None
    label = _SEVERITY_LABELS.get(severity_int)
    if label is None:
        return "unknown", severity_int
    return label, severity_int


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


def format_diagnostics(
    diagnostics: List[Dict], select_line: int = -1, column: Optional[int] = None
) -> List[str]:
    """Format diagnostics for legacy text responses."""

    if select_line != -1:
        diagnostics = _filter_diagnostics(diagnostics, select_line, column)

    formatted_messages: List[str] = []
    for diag in diagnostics:
        severity_value = diag.get("severity")
        severity_label = _SEVERITY_LABELS.get(severity_value)
        severity_display = (
            severity_label.capitalize() if severity_label else str(severity_value)
        )
        range_dict = diag.get("fullRange", diag.get("range"))
        location_label = _format_range_label(range_dict, diag)

        code = diag.get("code")
        provenance_parts = [str(code)] if code else []
        provenance_suffix = f" ({'#'.join(provenance_parts)})" if provenance_parts else ""

        header = f"[{severity_display}] {location_label}{provenance_suffix}"

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
    return clean_rendered(goal)


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


def extract_range(content: str, range_info: Mapping[str, Mapping[str, int]]) -> str:
    """Extract the text from the content based on the range.

    Args:
        content (str): The content to extract from.
        range_info (Mapping[str, Mapping[str, int]]): The range to extract.

    Returns:
        str: The extracted range text.
    """
    try:
        start = range_info["start"]
        end = range_info["end"]
        start_line = int(start["line"])
        start_char = int(start["character"])
        end_line = int(end["line"])
        end_char = int(end["character"])
    except (KeyError, TypeError, ValueError):
        return "Range out of bounds"

    if min(start_line, start_char, end_line, end_char) < 0:
        return "Range out of bounds"

    segments = content.splitlines(keepends=True)
    line_starts: List[int] = []
    line_bodies: List[str] = []
    offset = 0

    if segments:
        for segment in segments:
            line_starts.append(offset)
            line_body = segment.rstrip("\r\n")
            line_bodies.append(line_body)
            offset += len(segment)
    else:
        line_starts.append(0)
        line_bodies.append("")
        offset = 0

    sentinel_index = len(line_starts)
    document_end = len(content)

    def _utf16_to_codepoint_index(line_body: str, utf16_index: int) -> Optional[int]:
        if utf16_index <= 0:
            return 0

        code_units_consumed = 0
        for idx, ch in enumerate(line_body):
            # Code points above the BMP occupy two UTF-16 code units.
            units = 2 if ord(ch) >= 0x10000 else 1
            next_units = code_units_consumed + units
            if utf16_index < next_units:
                return idx
            code_units_consumed = next_units
            if utf16_index == code_units_consumed:
                return idx + 1

        # If requested position is beyond line end, signal out-of-bounds.
        return None

    def resolve_position(line: int, character: int) -> Optional[int]:
        if line < 0 or character < 0:
            return None
        if line == sentinel_index:
            return document_end if character == 0 else None
        if line >= sentinel_index:
            return None
        line_body = line_bodies[line]
        codepoint_index = _utf16_to_codepoint_index(line_body, character)
        if codepoint_index is None:
            return None
        return line_starts[line] + codepoint_index

    start_offset = resolve_position(start_line, start_char)
    end_offset = resolve_position(end_line, end_char)

    if start_offset is None or end_offset is None or end_offset < start_offset:
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
            return {"line": line_number + 1, "column": char_index + 1}
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
    return _filter_diagnostics(diagnostics, line, column)


class OptionalTokenVerifier(TokenVerifier):
    def __init__(self, expected_token: str):
        self.expected_token = expected_token

    async def verify_token(self, token: str) -> AccessToken | None:
        if token == self.expected_token:
            return AccessToken(token=token, client_id="lean-lsp-mcp", scopes=["user"])
        return None
