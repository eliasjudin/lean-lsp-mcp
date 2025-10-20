"""Utilities for building uniform MCP responses with Markdown summaries."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple


CHARACTER_LIMIT = 25_000
DEFAULT_RESPONSE_FORMAT = "markdown"
JSON_RESPONSE_FORMAT = "json"
_VALID_RESPONSE_FORMATS = {DEFAULT_RESPONSE_FORMAT, JSON_RESPONSE_FORMAT}


def normalize_response_format(value: Optional[str]) -> str:
    """Return a normalised response format string with a Markdown default."""

    if value is None:
        return DEFAULT_RESPONSE_FORMAT

    normalized = value.strip().lower()
    if not normalized:
        return DEFAULT_RESPONSE_FORMAT

    if normalized in _VALID_RESPONSE_FORMATS:
        return normalized

    return DEFAULT_RESPONSE_FORMAT

_TRUNCATION_MARKDOWN = (
    "_Note: Output truncated to {limit:,} characters. Refine filters, narrow ranges, "
    "or paginate to retrieve additional data._"
)

_TRUNCATION_PLAIN = (
    "Output truncated to {limit:,} characters. Refine filters, narrow ranges, "
    "or paginate to retrieve additional data."
)


@dataclass
class _TextTarget:
    holder: MutableMapping[str, Any]
    field: str
    label: str


def build_markdown_summary(summary: str, details: Sequence[str] | None = None) -> str:
    """Return a Markdown summary block."""

    headline = summary.strip() or "(no summary provided)"
    lines = [f"**Summary:** {headline}"]

    if details:
        for detail in details:
            detail_text = (detail or "").strip()
            if detail_text:
                lines.append(f"- {detail_text}")

    return "\n".join(lines)


def _collect_text_targets(
    items: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[_TextTarget]]:
    processed: List[Dict[str, Any]] = [copy.deepcopy(item) for item in items]
    targets: List[_TextTarget] = []
    first_text_seen = False

    for index, item in enumerate(processed):
        item_type = item.get("type")
        if item_type == "text":
            text_value = item.get("text")
            if isinstance(text_value, str):
                label = "summary" if not first_text_seen else f"text[{index}]"
                first_text_seen = True
                targets.append(_TextTarget(item, "text", label))
        elif item_type == "resource":
            resource = item.get("resource")
            if isinstance(resource, MutableMapping):
                text_value = resource.get("text")
                mime_type = resource.get("mimeType")
                if isinstance(text_value, str) and mime_type != "application/json":
                    label = str(resource.get("uri") or f"resource[{index}]")
                    targets.append(_TextTarget(resource, "text", label))

    return processed, targets


def apply_character_limit(
    items: Sequence[Dict[str, Any]],
    *,
    limit: int = CHARACTER_LIMIT,
) -> Tuple[List[Dict[str, Any]], bool, List[str]]:
    """Trim text-bearing items so totals stay within the shared character cap."""

    processed, targets = _collect_text_targets(items)
    total_chars = sum(len(target.holder.get(target.field, "")) for target in targets)

    if total_chars <= limit:
        return processed, False, []

    markdown_hint = _TRUNCATION_MARKDOWN.format(limit=limit)
    allowed_text = max(0, limit - len(markdown_hint))
    reduction_needed = total_chars - allowed_text

    truncated_labels: List[str] = []
    for target in reversed(targets):
        if reduction_needed <= 0:
            break
        text_value = target.holder.get(target.field)
        if not isinstance(text_value, str) or not text_value:
            continue
        remove = min(len(text_value), reduction_needed)
        new_length = len(text_value) - remove
        target.holder[target.field] = text_value[:new_length].rstrip()
        reduction_needed -= remove
        truncated_labels.append(target.label)

    truncated_labels = list(dict.fromkeys(reversed(truncated_labels)))

    if reduction_needed > 0 and markdown_hint:
        markdown_hint = markdown_hint[: max(0, len(markdown_hint) - reduction_needed)]

    processed.append({"type": "text", "text": markdown_hint})
    return processed, True, truncated_labels


def extend_structured_with_truncation(
    structured: Optional[Mapping[str, Any]],
    *,
    truncated: bool,
    truncated_sections: Sequence[str],
    limit: int = CHARACTER_LIMIT,
) -> Optional[Dict[str, Any]]:
    """Return a structured payload enriched with truncation metadata."""

    if structured is None and not truncated:
        return None

    if structured is None:
        structured_payload: Dict[str, Any] = {}
    elif isinstance(structured, dict):
        structured_payload = copy.deepcopy(structured)
    else:
        structured_payload = copy.deepcopy(dict(structured))

    if truncated:
        meta = structured_payload.setdefault("_meta", {})
        meta["truncated"] = True
        meta["character_limit"] = limit
        meta["truncation_hint"] = _TRUNCATION_PLAIN.format(limit=limit)
        if truncated_sections:
            meta["truncated_sections"] = list(truncated_sections)

    return structured_payload


__all__ = [
    "CHARACTER_LIMIT",
    "DEFAULT_RESPONSE_FORMAT",
    "JSON_RESPONSE_FORMAT",
    "normalize_response_format",
    "apply_character_limit",
    "build_markdown_summary",
    "extend_structured_with_truncation",
]
