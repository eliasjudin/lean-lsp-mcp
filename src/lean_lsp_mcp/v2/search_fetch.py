from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from urllib.parse import quote

from leanclient import LeanLSPClient

from lean_lsp_mcp.file_utils import get_file_contents, get_relative_file_path
from lean_lsp_mcp.models import LocalSearchResults
from lean_lsp_mcp.utils import LeanToolError, get_declaration_range
from lean_lsp_mcp.v2.contracts import FetchPayload, SearchPayload, SearchResultDoc


def encode_declaration_id(path: str, symbol: str) -> str:
    payload = {"path": path, "symbol": symbol}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_declaration_id(identifier: str) -> dict[str, str]:
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

    return {"path": path, "symbol": symbol}


def build_canonical_url(identifier: str) -> str:
    base = os.environ.get("LEAN_PUBLIC_BASE_URL", "").strip()
    if base:
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
        identifier = encode_declaration_id(path=relative_path, symbol=item.name)
        docs.append(
            SearchResultDoc(
                id=identifier,
                title=item.name,
                url=build_canonical_url(identifier),
            )
        )
    return SearchPayload(results=docs)


def declaration_text_for_id(
    *,
    workspace_root: Path,
    client: LeanLSPClient,
    identifier: str,
) -> FetchPayload:
    decoded = decode_declaration_id(identifier)
    rel_path = decoded["path"]
    symbol = decoded["symbol"]

    abs_path = (workspace_root / rel_path).resolve()
    if not abs_path.exists():
        raise LeanToolError(f"Declaration file does not exist: {rel_path}")

    rel_to_project = get_relative_file_path(workspace_root, str(abs_path))
    if not rel_to_project:
        raise LeanToolError("Declaration id path is outside workspace root.")

    # Request declaration range via LSP, then extract line range.
    decl_range = get_declaration_range(client, rel_to_project, symbol)
    content = get_file_contents(str(abs_path))

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
        url=build_canonical_url(identifier),
        metadata={"path": rel_path, "symbol": symbol},
    )
