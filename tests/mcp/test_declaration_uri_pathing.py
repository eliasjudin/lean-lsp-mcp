from __future__ import annotations

from urllib.parse import urlparse

import pytest

from lean_lsp_mcp.tools_read import _normalize_file_uri_path, declaration_uri_to_path
from lean_lsp_mcp.utils import LeanToolError


def test_normalize_file_uri_path_windows_drive() -> None:
    parsed = urlparse("file:///C:/Users/alice/Foo.lean")
    assert _normalize_file_uri_path(parsed) == "C:/Users/alice/Foo.lean"


def test_normalize_file_uri_path_windows_drive_localhost() -> None:
    parsed = urlparse("file://localhost/C:/Users/alice/Foo.lean")
    assert _normalize_file_uri_path(parsed) == "C:/Users/alice/Foo.lean"


def test_normalize_file_uri_path_posix() -> None:
    parsed = urlparse("file:///home/alice/Foo.lean")
    assert _normalize_file_uri_path(parsed) == "/home/alice/Foo.lean"


def test_normalize_file_uri_path_unc() -> None:
    parsed = urlparse("file://server/share/Foo.lean")
    assert _normalize_file_uri_path(parsed) == "//server/share/Foo.lean"


def test_declaration_uri_to_path_rejects_unsupported_scheme() -> None:
    with pytest.raises(LeanToolError, match="Unsupported declaration URI scheme"):
        declaration_uri_to_path("https://example.com/Foo.lean")
