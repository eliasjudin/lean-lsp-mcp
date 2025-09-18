from __future__ import annotations

import importlib.util
from pathlib import Path

from conftest import ensure_mcp_stub

ensure_mcp_stub()

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "lean_lsp_mcp" / "utils.py"
spec = importlib.util.spec_from_file_location("lean_lsp_mcp.utils", MODULE_PATH)
utils = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(utils)


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
