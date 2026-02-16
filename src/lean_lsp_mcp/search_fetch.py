from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import re
from urllib.parse import quote, urlparse

from leanclient import LeanLSPClient

from lean_lsp_mcp.file_utils import get_file_contents, get_relative_file_path
from lean_lsp_mcp.models import LocalSearchResults
from lean_lsp_mcp.utils import LeanToolError, get_declaration_range
from lean_lsp_mcp.contracts import FetchPayload, SearchPayload, SearchResultDoc


_DECLARATION_START_PATTERN = re.compile(
    r"^\s*(?:theorem|lemma|def|axiom|class|instance|structure|inductive|abbrev|opaque)\b"
)
_LOCAL_HTTP_HOSTS = {"127.0.0.1", "localhost", "::1"}
_DEFAULT_CANONICAL_URL_SCHEME = "decl_path"
_DEFAULT_LEAN4WEB_BASE_URL = "https://live.lean-lang.org/"
_DEFAULT_LEAN4WEB_PROJECT = "mathlib-v4.24.0"


def _canonical_url_scheme() -> str:
    raw = os.environ.get(
        "LEAN_CANONICAL_URL_SCHEME", _DEFAULT_CANONICAL_URL_SCHEME
    ).strip()
    scheme = (raw or _DEFAULT_CANONICAL_URL_SCHEME).lower()
    if scheme in {"decl_path", "decl"}:
        return "decl_path"
    if scheme in {"lean4web", "live"}:
        return "lean4web"
    raise LeanToolError(
        "LEAN_CANONICAL_URL_SCHEME must be one of: decl_path, lean4web."
    )


def _validated_http_base_url(*, value: str, env_name: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LeanToolError(f"{env_name} must be an absolute HTTP(S) URL.")
    if parsed.scheme == "http" and parsed.hostname not in _LOCAL_HTTP_HOSTS:
        raise LeanToolError(
            f"{env_name} must use HTTPS (HTTP is only allowed for localhost testing)."
        )
    return value


def _render_live_code(
    *,
    symbol: str | None,
    rel_path: str | None,
    declaration_text: str | None,
) -> str:
    lines = ["import Mathlib", ""]
    if rel_path:
        lines.append(f"-- source: {rel_path}")

    if declaration_text:
        lines.append(declaration_text.rstrip())
    elif symbol:
        lines.append(f"#check {symbol}")
    else:
        lines.append("-- declaration lookup")

    return "\n".join(lines).rstrip() + "\n"


def _encode_live_code(code: str) -> tuple[str, str]:
    mode_raw = os.environ.get("LEAN_LIVE_CODE_PARAM", "code").strip().lower()
    mode = mode_raw or "code"
    if mode == "code":
        return "code", quote(code, safe="")
    if mode == "codez":
        try:
            from lzstring import LZString  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise LeanToolError(
                "LEAN_LIVE_CODE_PARAM=codez requires the optional 'lzstring' package."
            ) from exc

        compressed = LZString().compressToBase64(code).rstrip("=")
        return "codez", quote(compressed, safe="")

    raise LeanToolError("LEAN_LIVE_CODE_PARAM must be one of: code, codez.")


def _build_lean4web_url(
    *,
    identifier: str,
    symbol: str | None,
    rel_path: str | None,
    declaration_text: str | None,
) -> str:
    base_raw = (
        os.environ.get("LEAN_LIVE_BASE_URL", "").strip() or _DEFAULT_LEAN4WEB_BASE_URL
    )
    base = _validated_http_base_url(value=base_raw, env_name="LEAN_LIVE_BASE_URL")
    parsed_base = urlparse(base)
    base_without_hash = parsed_base._replace(query="", fragment="").geturl()

    decoded_symbol = symbol
    decoded_path = rel_path
    if decoded_symbol is None or decoded_path is None:
        try:
            decoded = decode_declaration_id(identifier)
            if decoded_symbol is None and isinstance(decoded.get("symbol"), str):
                decoded_symbol = decoded["symbol"]
            if decoded_path is None and isinstance(decoded.get("path"), str):
                decoded_path = decoded["path"]
        except LeanToolError:
            pass

    code = _render_live_code(
        symbol=decoded_symbol,
        rel_path=decoded_path,
        declaration_text=declaration_text,
    )
    code_key, encoded_code = _encode_live_code(code)
    project_raw = (
        os.environ.get("LEAN_LIVE_PROJECT", "").strip() or _DEFAULT_LEAN4WEB_PROJECT
    )
    project = quote(project_raw, safe="")

    return f"{base_without_hash}#project={project}&{code_key}={encoded_code}"


def _leading_indent_width(line: str) -> int:
    return len(line) - len(line.lstrip())


def encode_declaration_id(path: str, symbol: str, line: int | None = None) -> str:
    payload = {"path": path, "symbol": symbol}
    if line is not None:
        if line < 1:
            raise LeanToolError("Declaration line must be >= 1.")
        payload["line"] = line
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_declaration_id(identifier: str) -> dict[str, str | int | None]:
    padding = "=" * ((4 - len(identifier) % 4) % 4)
    try:
        raw = base64.urlsafe_b64decode((identifier + padding).encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise LeanToolError("Invalid declaration id.") from exc

    path = payload.get("path")
    symbol = payload.get("symbol")
    if not isinstance(path, str) or not isinstance(symbol, str):
        raise LeanToolError("Invalid declaration id payload.")

    raw_line = payload.get("line")
    if raw_line is None:
        line: int | None = None
    elif isinstance(raw_line, int) and raw_line >= 1:
        line = raw_line
    else:
        raise LeanToolError("Invalid declaration id payload.")

    return {"path": path, "symbol": symbol, "line": line}


def build_canonical_url(
    identifier: str,
    *,
    symbol: str | None = None,
    rel_path: str | None = None,
    declaration_text: str | None = None,
) -> str:
    if _canonical_url_scheme() == "lean4web":
        return _build_lean4web_url(
            identifier=identifier,
            symbol=symbol,
            rel_path=rel_path,
            declaration_text=declaration_text,
        )

    base = (
        os.environ.get("LEAN_PUBLIC_BASE_URL", "").strip()
        or os.environ.get("LEAN_OAUTH_RESOURCE_SERVER_URL", "").strip()
    )
    if base:
        _validated_http_base_url(
            value=base,
            env_name="LEAN_PUBLIC_BASE_URL (or LEAN_OAUTH_RESOURCE_SERVER_URL)",
        )
        return f"{base.rstrip('/')}/decl/{quote(identifier)}"
    return f"lean://decl/{identifier}"


def _workspace_relative_if_fetchable(workspace_root: Path, path: str) -> str | None:
    root = workspace_root.resolve()
    raw = Path(path)
    candidate = raw if raw.is_absolute() else (root / raw)

    try:
        resolved = candidate.resolve()
    except OSError:
        return None

    if not resolved.exists() or not resolved.is_file():
        return None

    try:
        rel = resolved.relative_to(root)
    except ValueError:
        return None

    return str(rel)


def search_payload_from_local_results(
    results: LocalSearchResults,
    workspace_root: Path,
) -> SearchPayload:
    docs = []
    for item in results.items:
        relative_path = _workspace_relative_if_fetchable(workspace_root, item.file)
        if relative_path is None:
            continue
        identifier = encode_declaration_id(
            path=relative_path,
            symbol=item.name,
            line=item.line,
        )
        docs.append(
            SearchResultDoc(
                id=identifier,
                title=item.name,
                url=build_canonical_url(
                    identifier, symbol=item.name, rel_path=relative_path
                ),
            )
        )
    return SearchPayload(results=docs)


def _extract_declaration_text_from_line(*, content: str, line: int) -> str:
    lines = content.splitlines()
    if not lines:
        raise LeanToolError("Declaration source file is empty.")

    if line < 1 or line > len(lines):
        raise LeanToolError("Declaration id line is out of range.")

    start_idx = line - 1
    lookahead_end = min(len(lines), start_idx + 6)
    resolved_start = None
    for idx in range(start_idx, lookahead_end):
        if _DECLARATION_START_PATTERN.match(lines[idx]):
            resolved_start = idx
            break

    if resolved_start is None:
        raise LeanToolError("Could not resolve declaration start line from fetch id.")

    base_indent = _leading_indent_width(lines[resolved_start])
    end_idx = len(lines)
    for idx in range(resolved_start + 1, len(lines)):
        if not _DECLARATION_START_PATTERN.match(lines[idx]):
            continue
        # Stop only at same-or-less indentation than the declaration start.
        # More-indented declaration starters can appear in local `where` blocks.
        if _leading_indent_width(lines[idx]) <= base_indent:
            end_idx = idx
            break

    selected_lines = lines[resolved_start:end_idx]
    while selected_lines and not selected_lines[-1].strip():
        selected_lines.pop()

    selected = "\n".join(selected_lines)
    if not selected.strip():
        raise LeanToolError("Resolved declaration text is empty.")
    return selected


def declaration_text_for_id(
    *,
    workspace_root: Path,
    client: LeanLSPClient | None,
    identifier: str,
) -> FetchPayload:
    decoded = decode_declaration_id(identifier)
    rel_path_raw = decoded["path"]
    symbol_raw = decoded["symbol"]
    line_raw = decoded["line"]
    if not isinstance(rel_path_raw, str) or not isinstance(symbol_raw, str):
        raise LeanToolError("Invalid declaration id payload.")

    rel_path = rel_path_raw
    symbol = symbol_raw
    line = line_raw if isinstance(line_raw, int) else None

    abs_path = (workspace_root / rel_path).resolve()
    if not abs_path.exists():
        raise LeanToolError(f"Declaration file does not exist: {rel_path}")

    rel_to_project = get_relative_file_path(workspace_root, str(abs_path))
    if not rel_to_project:
        raise LeanToolError("Declaration id path is outside workspace root.")

    content = get_file_contents(str(abs_path))

    if line is not None:
        selected = _extract_declaration_text_from_line(content=content, line=line)
        return FetchPayload(
            id=identifier,
            title=symbol,
            text=selected,
            url=build_canonical_url(
                identifier,
                symbol=symbol,
                rel_path=rel_path,
                declaration_text=selected,
            ),
            metadata={"path": rel_path, "symbol": symbol, "line": str(line)},
        )

    if client is None:
        raise LeanToolError("Fetch id requires Lean client initialization.")

    # Legacy IDs do not include a line anchor; resolve via LSP declaration range.
    decl_range = get_declaration_range(client, rel_to_project, symbol)
    if decl_range is None:
        raise LeanToolError(
            "Could not resolve declaration range for fetch id; refusing full-file fallback."
        )
    start_line, end_line = decl_range
    lines = content.splitlines()
    start_idx = max(start_line - 1, 0)
    end_idx = min(end_line, len(lines))
    selected = "\n".join(lines[start_idx:end_idx])

    return FetchPayload(
        id=identifier,
        title=symbol,
        text=selected,
        url=build_canonical_url(
            identifier,
            symbol=symbol,
            rel_path=rel_path,
            declaration_text=selected,
        ),
        metadata={"path": rel_path, "symbol": symbol},
    )
