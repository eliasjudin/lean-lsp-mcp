from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, AliasChoices, field_validator


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


class LeanFilePathInput(ToolInputBase):
    file_path: str = Field(
        ...,
        description="Absolute or project-relative path to a Lean source file.",
    )

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
        description="1-based column within the line; omit to auto-detect goal.",
    )


class LeanStateSearchInput(LeanFileLocationInput):
    column: int = Field(
        ...,
        ge=1,
        description="1-based column within the line to query for goals.",
    )
    num_results: int = Field(
        default=5,
        ge=1,
        description="Maximum number of matching suggestions to return.",
    )


class LeanHammerPremiseInput(LeanStateSearchInput):
    num_results: int = Field(
        default=32,
        ge=1,
        description="Maximum number of premises to retrieve from the hammer.",
    )


class LeanSearchInput(ToolInputBase):
    query: str = Field(
        ...,
        description="Search query passed to leansearch.net.",
        min_length=1,
    )
    num_results: int = Field(
        default=5,
        ge=1,
        description="Maximum number of results to fetch from the service.",
    )


class LoogleSearchInput(ToolInputBase):
    query: str = Field(
        ...,
        description="Search query sent to loogle.lean-lang.org.",
        min_length=1,
    )
    num_results: int = Field(
        default=8,
        ge=1,
        description="Maximum number of results to include in the response.",
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
        description="1-based column to inspect for hover information.",
    )


class LeanCompletionsInput(LeanFileLocationInput):
    column: int = Field(
        ...,
        ge=1,
        description="1-based column where completions are requested.",
    )
    max_completions: int = Field(
        default=32,
        ge=1,
        description="Maximum number of completion items to include in the response.",
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
