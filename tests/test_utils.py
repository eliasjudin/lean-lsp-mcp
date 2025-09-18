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
