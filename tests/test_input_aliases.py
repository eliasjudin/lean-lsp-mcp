from __future__ import annotations

import os
from urllib.parse import urlparse, unquote

from conftest import load_from_src


tool_inputs = load_from_src("lean_lsp_mcp.tool_inputs")


def _expected_path_from_file_uri(uri: str) -> str:
    parsed = urlparse(uri)
    path = unquote(parsed.path or "")
    if os.name == "nt" and path.startswith("/"):
        path = path.lstrip("/")
    return path


def test_diagnostics_accepts_uri_alias():
    uri = "file:///tmp/Example.lean"
    params = {"uri": uri}
    model = tool_inputs.LeanDiagnosticMessagesInput(**params)

    assert isinstance(model.file_path, str) and model.file_path
    assert model.file_path == _expected_path_from_file_uri(uri)


def test_diagnostics_accepts_nested_file_object():
    # When both `path` and `uri` are present, prefer `path` for file_path
    params = {"file": {"path": "src/Main.lean", "uri": "file:///tmp/Main.lean"}}
    model = tool_inputs.LeanDiagnosticMessagesInput(**params)

    assert model.file_path == "src/Main.lean"

