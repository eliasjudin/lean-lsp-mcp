from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, unquote

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ToolInputBase(BaseModel):
    """Shared configuration for structured tool inputs."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        populate_by_name=True,
        extra="forbid",
    )

    response_format: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("_format", "response_format"),
        serialization_alias="_format",
        description="Optional response rendering hint (legacy key `_format`).",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_lsp_position(cls, data: Any) -> Any:
        """Accept an LSP-style nested position and normalize to 1-based.

        Allows inputs like::
            {"position": {"line": 0, "character": 4}}

        and rewrites them to top-level 1-based ``line``/``column`` fields, removing
        the ``position`` key to satisfy ``extra='forbid'``.
        """
        if not isinstance(data, dict):
            return data

        pos: Optional[Dict[str, Any]] = data.get("position")
        if isinstance(pos, dict):
            # Determine semantics: if `character` is present (LSP), treat as 0-based.
            # If `column` is present (convenience), treat as 1-based.
            has_char = "character" in pos
            has_col = "column" in pos

            # Only set if not already provided explicitly at the top level.
            if "line" in pos and "line" not in data:
                try:
                    line_val = int(pos["line"]) if pos["line"] is not None else None
                    if line_val is not None:
                        data["line"] = line_val + 1 if (has_char and not has_col) else line_val
                except Exception:
                    pass

            # Column handling: prefer LSP `character` when present; otherwise accept `column`.
            if "column" not in data and "character" not in data:
                try:
                    if has_char:
                        data["column"] = int(pos["character"]) + 1
                    elif has_col:
                        data["column"] = int(pos["column"])  # already 1-based
                except Exception:
                    pass

            # Drop nested key to avoid extra='forbid'.
            data = {k: v for k, v in data.items() if k != "position"}

        return data


class LeanFilePathInput(ToolInputBase):
    file_path: str = Field(
        ...,
        # Accept common variants the LLMs often use (`uri`, `path`).
        validation_alias=AliasChoices("file_path", "uri", "path"),
        description="Absolute or project-relative path to a Lean source file.",
    )

    @model_validator(mode="before")
    @classmethod
    def _promote_nested_file_keys(cls, data: Any) -> Any:
        """Accept nested `file` payloads by promoting to top-level.

        Many tools return a structured `file` object with `uri`/`path`; allow
        callers to pass that back directly by copying into top-level fields and
        dropping the nested key to satisfy `extra='forbid'`.
        """
        if isinstance(data, dict):
            f = data.get("file")
            if isinstance(f, dict):
                # Only set if not already present at top-level
                if "file_path" not in data:
                    if "path" in f and f.get("path"):
                        data["file_path"] = f.get("path")
                    elif "uri" in f and f.get("uri"):
                        # Set as file_path; a later validator will coerce file:// URIs
                        data["file_path"] = f.get("uri")
                # Remove nested object to avoid extra='forbid'
                data = {k: v for k, v in data.items() if k != "file"}
        return data

    @field_validator("file_path", mode="before")
    @classmethod
    def _coerce_file_uri(cls, value: Any) -> Any:
        """Allow `file://` URIs by converting to a local path string."""
        if isinstance(value, str) and value.startswith("file://"):
            parsed = urlparse(value)
            path = unquote(parsed.path or "")
            if os.name == "nt":
                # Preserve UNC shares: file://server/share -> \\server\share
                if parsed.netloc:
                    unc = f"\\\\{parsed.netloc}{path.replace('/', '\\')}"
                    return unc
                # Drive-letter paths often include a leading slash (e.g., /C:/...)
                if path.startswith("/"):
                    path = path.lstrip("/")
                return path
            return path
        return value

    @field_validator("file_path")
    @classmethod
    def _validate_file_path(cls, value: str) -> str:
        if not value:
            raise ValueError("file_path cannot be empty.")
        return value


class LeanFileContentsInput(LeanFilePathInput):
    annotate_lines: bool = Field(
        default=True,
        description="Annotate each returned line with its 1-based line number.",
    )
    start_line: Optional[int] = Field(
        default=None,
        ge=1,
        description="1-based line number to start reading from.",
    )
    line_count: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of lines to return from `start_line`.",
    )


class LeanFileLocationInput(LeanFilePathInput):
    line: int = Field(
        ...,
        ge=1,
        description="1-based line number within the Lean source file.",
    )


class LeanGoalInput(LeanFileLocationInput):
    column: Optional[int] = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("column", "character"),
        description="1-based column within the line; omit to auto-detect goal.",
    )


class LeanStateSearchInput(LeanFileLocationInput):
    column: int = Field(
        ...,
        ge=1,
        validation_alias=AliasChoices("column", "character"),
        description="1-based column within the line to query for goals.",
    )
    num_results: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of matching suggestions to return (1-100).",
    )


class LeanHammerPremiseInput(LeanStateSearchInput):
    num_results: int = Field(
        default=32,
        ge=1,
        le=100,
        description="Maximum number of premises to retrieve from the hammer (1-100).",
    )


class LeanSearchInput(ToolInputBase):
    query: str = Field(
        ...,
        description="Search query passed to leansearch.net.",
        min_length=1,
        max_length=500,
    )
    num_results: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of results to fetch from the service (1-50).",
    )


class LoogleSearchInput(ToolInputBase):
    query: str = Field(
        ...,
        description="Search query sent to loogle.lean-lang.org.",
        min_length=1,
        max_length=500,
    )
    num_results: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Maximum number of results to include in the response (1-50).",
    )


class LeanBuildInput(ToolInputBase):
    lean_project_path: Optional[str] = Field(
        default=None,
        description="Absolute path to the Lean project root; inferred when omitted.",
    )
    clean: bool = Field(
        default=False,
        description="Whether to run `lake clean` before rebuilding the project.",
    )


class LeanDiagnosticMessagesInput(LeanFilePathInput):
    start_line: Optional[int] = Field(
        default=None,
        ge=1,
        description="1-based line number to start collecting diagnostics from.",
    )
    line_count: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of lines to scan for diagnostics from `start_line`.",
    )


class LeanTermGoalInput(LeanGoalInput):
    """Input for `lean_term_goal`; inherits goal position semantics."""


class LeanHoverInput(LeanFileLocationInput):
    column: int = Field(
        ...,
        ge=1,
        validation_alias=AliasChoices("column", "character"),
        description="1-based column to inspect for hover information.",
    )


class LeanCompletionsInput(LeanFileLocationInput):
    column: int = Field(
        ...,
        ge=1,
        validation_alias=AliasChoices("column", "character"),
        description="1-based column where completions are requested.",
    )
    max_completions: int = Field(
        default=32,
        ge=1,
        le=100,
        description="Maximum number of completion items to include in the response (1-100).",
    )


class LeanDeclarationFileInput(LeanFilePathInput):
    symbol: str = Field(
        ...,
        min_length=1,
        description="Name of the symbol to locate within the file.",
    )


class LeanMultiAttemptInput(LeanFileLocationInput):
    snippets: List[str] = Field(
        ...,
        min_length=1,
        description="One or more Lean code snippets to try at the target line.",
    )


class LeanRunCodeInput(ToolInputBase):
    code: str = Field(
        ...,
        min_length=1,
        description="Complete Lean snippet to execute in an isolated buffer.",
    )


class LeanToolSpecInput(ToolInputBase):
    """Input for `lean_tool_spec`; accepts optional formatting hints only."""

    pass
