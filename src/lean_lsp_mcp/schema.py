"""Shared response schema helpers for Lean LSP MCP.

Simplified to minimize response size:
- No global schema version in responses.
- Omit the ``meta`` field entirely when empty.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, MutableMapping

from lean_lsp_mcp.schema_types import ResponseMeta

RESPONSE_FORMAT_ENV = "LEAN_LSP_MCP_RESPONSE_FORMAT"

LegacyFormatter = Callable[[Dict[str, Any]], Any] | str | None


def _is_legacy_mode() -> bool:
    return os.getenv(RESPONSE_FORMAT_ENV, "structured").lower() == "legacy"


def make_response(
    status: str,
    data: Any = None,
    meta: MutableMapping[str, Any] | ResponseMeta | None = None,
    legacy_formatter: LegacyFormatter = None,
) -> Any:
    """Create a response envelope or fall back to legacy formatting.

    Args:
        status: Either "ok" or "error".
        data: Tool specific payload.
        meta: Optional metadata to attach.
        legacy_formatter: Callable that receives the structured response and
            returns the legacy payload (typically a string). If ``None`` and
            legacy mode is requested the raw data is returned.
    """

    # Build minimal envelope and only include meta if provided.
    envelope: Dict[str, Any] = {"status": status, "data": data}
    if meta:
        # Avoid inserting empty meta objects to reduce tokens.
        # Copy to a regular dict to ensure JSON-serializable structure.
        response_meta: Dict[str, Any] = dict(meta)
        if response_meta:
            envelope["meta"] = response_meta

    if _is_legacy_mode():
        if legacy_formatter is None:
            return data
        if isinstance(legacy_formatter, str):
            return legacy_formatter
        return legacy_formatter(envelope)

    return envelope


__all__ = ["RESPONSE_FORMAT_ENV", "make_response"]
