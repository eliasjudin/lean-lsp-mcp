from __future__ import annotations

import os
from pathlib import Path

import pytest

from conftest import load_from_src

utils = load_from_src("lean_lsp_mcp.utils")


def test_normalize_range_preserves_zero_based_positions():
    original = {
        "start": {"line": 0, "character": 4},
        "end": {"line": 1, "character": 0},
    }
    normalized = utils.normalize_range(original)
    assert normalized == {
        "start": {"line": 0, "character": 4},
        "end": {"line": 1, "character": 0},
    }


def test_diagnostics_to_entries_handles_basic_fields():
    diagnostics = [
        {
            "message": "oops",
            "severity": 2,
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 5},
            },
            "source": "lean",
        }
    ]
    entries = utils.diagnostics_to_entries(diagnostics)
    assert entries[0]["message"] == "oops"
    assert entries[0]["severity"] == "warning"
    assert entries[0]["severityCode"] == 2
    assert entries[0]["range"]["start"] == {"line": 0, "character": 0}


def test_summarize_diagnostics_counts_by_severity():
    summary = utils.summarize_diagnostics(
        [
            {"severity": "error"},
            {"severity": "warning"},
            {"severity": "warning"},
            {"severity": "hint"},
        ]
    )
    assert summary["count"] == 4
    assert summary["bySeverity"] == {
        "error": 1,
        "warning": 2,
        "hint": 1,
    }
    assert summary["has_errors"] is True


def test_goal_to_payload_extracts_rendered_text():
    goal = {
        "rendered": "```lean\ngoal text\n```",
        "goals": ["goal text"],
        "userState": "state",
    }
    payload = utils.goal_to_payload(goal)
    assert payload["rendered"] == "goal text"
    assert payload["goals"] == ["goal text"]
    assert payload["user_state"] == "state"


def test_compute_pagination_returns_expected_slice():
    start, end, meta = utils.compute_pagination(100, 11, 10)
    assert (start, end) == (11, 20)
    assert meta["has_more"] is True
    assert meta["next_start_line"] == 21


def test_compute_pagination_defaults_to_full_range():
    start, end, meta = utils.compute_pagination(5, None, None)
    assert (start, end) == (1, 5)
    assert meta["has_more"] is False
    assert "next_start_line" not in meta


def test_file_identity_generates_sanitized_uri():
    rel_path = "Foo\\Bar/Proof.lean"
    abs_path = os.path.join(os.getcwd(), "Foo", "Bar", "Proof.lean")
    identity = utils.file_identity(rel_path, abs_path)
    assert identity["relative_path"] == "Foo/Bar/Proof.lean"
    assert identity["uri"] == Path(abs_path).resolve().as_uri()


def test_file_identity_without_absolute_path_omits_uri():
    identity = utils.file_identity("relative/path.lean")
    assert identity["relative_path"] == "relative/path.lean"
    assert identity["uri"] == ""


def test_uri_to_absolute_path_roundtrip(tmp_path):
    file_path = tmp_path / "Example With Space.lean"
    file_path.write_text("example", encoding="utf-8")
    uri = file_path.resolve().as_uri()

    resolved = utils.uri_to_absolute_path(uri)

    assert resolved == str(file_path.resolve())


def test_uri_to_absolute_path_rejects_non_file_scheme():
    assert utils.uri_to_absolute_path("https://example.com/foo") is None
    assert utils.uri_to_absolute_path(None) is None


def test_format_diagnostics_includes_severity_and_code():
    diagnostics = [
        {
            "message": "undefined identifier",
            "severity": 1,
            "range": {
                "start": {"line": 2, "character": 4},
                "end": {"line": 2, "character": 9},
            },
            "source": "elaborator",
            "code": "unknownId",
            "file": "src/Foo.lean",
        }
    ]

    formatted = utils.format_diagnostics(diagnostics)
    assert formatted == [
        "[Error] src/Foo.lean:3:5-3:10 (unknownId)\nundefined identifier"
    ]


def test_format_diagnostics_indents_related_information():
    diagnostics = [
        {
            "message": "unused variable",
            "severity": 2,
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 6},
            },
            "source": "linter",
            "code": "unused",
            "relatedInformation": [
                {
                    "message": "introduced here",
                    "location": {
                        "uri": "file:///tmp/Bar.lean",
                        "range": {
                            "start": {"line": 10, "character": 2},
                            "end": {"line": 10, "character": 5},
                        },
                    },
                }
            ],
        }
    ]

    formatted = utils.format_diagnostics(diagnostics)
    assert formatted[0].startswith("[Warning] 1:1-1:7 (unused)\nunused variable")
    assert "  /tmp/Bar.lean:11:3-11:6" in formatted[0]
    assert "  introduced here" in formatted[0]


def test_format_diagnostics_filters_by_line():
    diagnostics = [
        {
            "message": "first",
            "severity": 3,
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 1},
            },
        },
        {
            "message": "second",
            "severity": 3,
            "range": {
                "start": {"line": 5, "character": 0},
                "end": {"line": 5, "character": 1},
            },
        },
    ]

    formatted = utils.format_diagnostics(diagnostics, select_line=0)
    assert formatted == ["[Info] 1:1-1:2\nfirst"]


def test_format_diagnostics_handles_missing_range():
    diagnostics = [
        {
            "message": "internal error",
            "severity": 1,
        }
    ]

    formatted = utils.format_diagnostics(diagnostics)
    assert formatted == ["[Error] No range\ninternal error"]


def test_output_capture_restores_after_exception():
    class Boom(Exception):
        pass

    capture = utils.OutputCapture()

    with pytest.raises(Boom):
        with capture:
            print("hello from capture")
            raise Boom()

    assert "hello from capture" in capture.get_output()

    # stdout should still be usable after the context manager exits
    print("stdout restored")


def test_extract_range_handles_virtual_document_end_line():
    content = "first\nsecond\n"
    result = utils.extract_range(
        content,
        {
            "start": {"line": 1, "character": 0},
            "end": {"line": 2, "character": 0},
        },
    )
    assert result == "second\n"


def test_extract_range_rejects_invalid_column():
    content = "abc"
    result = utils.extract_range(
        content,
        {
            "start": {"line": 0, "character": 0},
            "end": {"line": 0, "character": 5},
        },
    )
    assert result == "Range out of bounds"


def test_extract_range_allows_cursor_at_document_end():
    content = "line\n"
    result = utils.extract_range(
        content,
        {
            "start": {"line": 1, "character": 0},
            "end": {"line": 1, "character": 0},
        },
    )
    assert result == ""
