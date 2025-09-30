from __future__ import annotations

from conftest import load_from_src

utils = load_from_src("lean_lsp_mcp.utils")


def test_normalize_range_converts_to_one_indexed():
    original = {
        "start": {"line": 0, "character": 4},
        "end": {"line": 1, "character": 0},
    }
    normalized = utils.normalize_range(original)
    assert normalized == {
        "start": {"line": 1, "column": 5},
        "end": {"line": 2, "column": 1},
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
    assert entries[0]["severity"] == 2
    assert entries[0]["range"]["start"] == {"line": 1, "column": 1}


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


def test_format_diagnostics_includes_severity_source_and_code():
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
        "[Error] src/Foo.lean:3:5-3:10 (elaborator#unknownId)\nundefined identifier"
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
    assert formatted[0].startswith("[Warning] 1:1-1:7 (linter#unused)\nunused variable")
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
