from __future__ import annotations

from pydantic import BaseModel, Field


class SearchResultDoc(BaseModel):
    id: str = Field(description="Stable declaration identifier")
    title: str = Field(description="Human-readable declaration title")
    url: str = Field(description="Canonical citation URL for this declaration")


class SearchPayload(BaseModel):
    results: list[SearchResultDoc] = Field(default_factory=list)


class FetchPayload(BaseModel):
    id: str = Field(description="Stable declaration identifier")
    title: str = Field(description="Declaration title")
    text: str = Field(description="Declaration source text")
    url: str = Field(description="Canonical citation URL")
    metadata: dict[str, str] | None = Field(
        default=None, description="Optional declaration metadata"
    )
