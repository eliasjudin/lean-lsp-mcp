from __future__ import annotations

import asyncio
import os

from leanclient import LeanLSPClient
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field
from typing import Annotated

from lean_lsp_mcp.http_client import HttpRequestError, request_json
from lean_lsp_mcp.loogle import loogle_remote
from lean_lsp_mcp.models import (
    LeanFinderResult,
    LeanFinderResults,
    LeanSearchResult,
    LeanSearchResults,
    LocalSearchResult,
    LocalSearchResults,
    LoogleResult,
    LoogleResults,
    PremiseResult,
    PremiseResults,
    StateSearchResult,
    StateSearchResults,
)
from lean_lsp_mcp.profiles import external_tools_enabled
from lean_lsp_mcp.rate_limit import rate_limited
from lean_lsp_mcp.search_utils import lean_local_search
from lean_lsp_mcp.tool_utils import resolve_and_prepare_file, safe_report_progress
from lean_lsp_mcp.utils import LeanToolError


@rate_limited("loogle", max_requests=3, per_seconds=30)
async def _loogle_remote_rate_limited(
    ctx: Context,
    query: str,
    num_results: int,
) -> LoogleResults:
    await safe_report_progress(ctx, progress=1, total=10, message="Awaiting loogle")
    result = await asyncio.to_thread(loogle_remote, query, num_results)
    if isinstance(result, str):
        raise LeanToolError(result)
    return LoogleResults(items=result)


def register_external_tools(
    mcp: FastMCP,
    *,
    rg_available: bool,
    rg_message: str,
    logger,
) -> None:
    @mcp.tool(
        "local_search",
        annotations=ToolAnnotations(
            title="Local Search",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def local_search(
        ctx: Context,
        query: Annotated[str, Field(description="Declaration name or prefix")],
        limit: Annotated[int, Field(description="Max matches", ge=1)] = 10,
    ) -> LocalSearchResults:
        """Use this when you need a fast local declaration-name lookup in the workspace."""
        if not rg_available:
            raise LeanToolError(rg_message)

        app_ctx = ctx.request_context.lifespan_context
        try:
            raw_results = await asyncio.to_thread(
                lean_local_search,
                query=query.strip(),
                limit=limit,
                project_root=app_ctx.workspace_root,
            )
        except RuntimeError as exc:
            raise LeanToolError(f"Search failed: {exc}") from exc

        items = [
            LocalSearchResult(
                name=r["name"],
                kind=r["kind"],
                file=r["file"],
                line=(
                    r["line"]
                    if isinstance(r.get("line"), int) and r["line"] >= 1
                    else None
                ),
            )
            for r in raw_results
        ]
        return LocalSearchResults(items=items)

    @mcp.tool(
        "leansearch",
        annotations=ToolAnnotations(
            title="LeanSearch",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    @rate_limited("leansearch", max_requests=3, per_seconds=30)
    async def leansearch(
        ctx: Context,
        query: Annotated[str, Field(description="Natural language or Lean term query")],
        num_results: Annotated[int, Field(description="Max results", ge=1)] = 5,
    ) -> LeanSearchResults:
        """Use this when you need remote natural-language search on leansearch.net."""
        if not external_tools_enabled("leansearch"):
            raise LeanToolError("leansearch is disabled by server configuration.")

        await safe_report_progress(
            ctx, progress=1, total=10, message="Awaiting leansearch.net"
        )
        try:
            results = await request_json(
                "POST",
                "https://leansearch.net/search",
                headers={"Content-Type": "application/json"},
                json_data={"num_results": str(num_results), "query": [query]},
                timeout=10,
            )
        except HttpRequestError as exc:
            raise LeanToolError(str(exc)) from exc

        if not results or not results[0]:
            return LeanSearchResults(items=[])

        raw_results = [r["result"] for r in results[0][:num_results]]
        items = [
            LeanSearchResult(
                name=".".join(r["name"]),
                module_name=".".join(r["module_name"]),
                kind=r.get("kind"),
                type=r.get("type"),
            )
            for r in raw_results
        ]
        return LeanSearchResults(items=items)

    @mcp.tool(
        "loogle",
        annotations=ToolAnnotations(
            title="Loogle",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def loogle(
        ctx: Context,
        query: Annotated[
            str, Field(description="Type pattern, constant, or name substring")
        ],
        num_results: Annotated[int, Field(description="Max results", ge=1)] = 8,
    ) -> LoogleResults:
        """Use this when you need type-signature search through loogle."""
        if not external_tools_enabled("loogle"):
            raise LeanToolError("loogle is disabled by server configuration.")

        app_ctx = ctx.request_context.lifespan_context

        if app_ctx.loogle_local_available and app_ctx.loogle_manager:
            if app_ctx.lean_project_path != app_ctx.loogle_manager.project_path:
                if app_ctx.loogle_manager.set_project_path(app_ctx.lean_project_path):
                    await app_ctx.loogle_manager.stop()
            try:
                results = await app_ctx.loogle_manager.query(query, num_results)
                items = [
                    LoogleResult(
                        name=r.get("name", ""),
                        type=r.get("type", ""),
                        module=r.get("module", ""),
                    )
                    for r in results
                ]
                return LoogleResults(items=items)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Local loogle failed: %s, falling back to remote", exc)

        return await _loogle_remote_rate_limited(
            ctx=ctx, query=query, num_results=num_results
        )

    @mcp.tool(
        "leanfinder",
        annotations=ToolAnnotations(
            title="Lean Finder",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    @rate_limited("leanfinder", max_requests=10, per_seconds=30)
    async def leanfinder(
        ctx: Context,
        query: Annotated[str, Field(description="Mathematical concept or proof state")],
        num_results: Annotated[int, Field(description="Max results", ge=1)] = 5,
    ) -> LeanFinderResults:
        """Use this when you need semantic mathlib retrieval from Lean Finder."""
        if not external_tools_enabled("leanfinder"):
            raise LeanToolError("leanfinder is disabled by server configuration.")

        request_url = (
            "https://bxrituxuhpc70w8w.us-east-1.aws.endpoints.huggingface.cloud"
        )

        await safe_report_progress(
            ctx, progress=1, total=10, message="Awaiting Lean Finder"
        )
        try:
            data = await request_json(
                "POST",
                request_url,
                headers={"Content-Type": "application/json"},
                json_data={"inputs": query, "top_k": int(num_results)},
                timeout=10,
            )
        except HttpRequestError as exc:
            raise LeanToolError(str(exc)) from exc

        import re

        results: list[LeanFinderResult] = []
        for result in data.get("results", []):
            if "https://leanprover-community.github.io/mathlib4_docs" not in result.get(
                "url", ""
            ):
                continue
            match = re.search(r"pattern=(.*?)#doc", result["url"])
            if match:
                results.append(
                    LeanFinderResult(
                        full_name=match.group(1),
                        formal_statement=result.get("formal_statement", ""),
                        informal_statement=result.get("informal_statement", ""),
                    )
                )

        return LeanFinderResults(items=results)

    @mcp.tool(
        "state_search",
        annotations=ToolAnnotations(
            title="State Search",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    @rate_limited("lean_state_search", max_requests=3, per_seconds=30)
    async def state_search(
        ctx: Context,
        path: Annotated[str, Field(description="Workspace-relative path to Lean file")],
        line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
        column: Annotated[int, Field(description="Column number (1-indexed)", ge=1)],
        num_results: Annotated[int, Field(description="Max results", ge=1)] = 5,
    ) -> StateSearchResults:
        """Use this when you need premise-search suggestions for the current proof goal."""
        if not external_tools_enabled("state_search"):
            raise LeanToolError("state_search is disabled by server configuration.")

        _, rel_path = resolve_and_prepare_file(ctx, path)
        client: LeanLSPClient = ctx.request_context.lifespan_context.client
        client.open_file(rel_path)
        goal = client.get_goal(rel_path, line - 1, column - 1)

        if not goal or not goal.get("goals"):
            raise LeanToolError(
                f"No goals found at line {line}, column {column}. Try a different position."
            )

        base_url = os.getenv(
            "LEAN_STATE_SEARCH_URL", "https://premise-search.com"
        ).rstrip("/")
        await safe_report_progress(
            ctx, progress=1, total=10, message=f"Awaiting {base_url}"
        )
        try:
            results = await request_json(
                "GET",
                f"{base_url}/api/search",
                params={
                    "query": goal["goals"][0],
                    "results": str(num_results),
                    "rev": "v4.22.0",
                },
                timeout=10,
            )
        except HttpRequestError as exc:
            raise LeanToolError(str(exc)) from exc

        items = [StateSearchResult(name=r["name"]) for r in results]
        return StateSearchResults(items=items)

    @mcp.tool(
        "hammer_premise",
        annotations=ToolAnnotations(
            title="Hammer Premises",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    @rate_limited("hammer_premise", max_requests=3, per_seconds=30)
    async def hammer_premise(
        ctx: Context,
        path: Annotated[str, Field(description="Workspace-relative path to Lean file")],
        line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
        column: Annotated[int, Field(description="Column number (1-indexed)", ge=1)],
        num_results: Annotated[int, Field(description="Max results", ge=1)] = 32,
    ) -> PremiseResults:
        """Use this when you need hammer premise candidates for an active goal."""
        if not external_tools_enabled("hammer_premise"):
            raise LeanToolError("hammer_premise is disabled by server configuration.")

        _, rel_path = resolve_and_prepare_file(ctx, path)
        client: LeanLSPClient = ctx.request_context.lifespan_context.client
        client.open_file(rel_path)
        goal = client.get_goal(rel_path, line - 1, column - 1)

        if not goal or not goal.get("goals"):
            raise LeanToolError(
                f"No goals found at line {line}, column {column}. Try a different position."
            )

        data = {"state": goal["goals"][0], "new_premises": [], "k": int(num_results)}
        base_url = os.getenv("LEAN_HAMMER_URL", "http://leanpremise.net").rstrip("/")

        await safe_report_progress(
            ctx, progress=1, total=10, message=f"Awaiting {base_url}"
        )
        try:
            results = await request_json(
                "POST",
                f"{base_url}/retrieve",
                headers={"Content-Type": "application/json"},
                json_data=data,
                timeout=10,
            )
        except HttpRequestError as exc:
            raise LeanToolError(str(exc)) from exc

        items = [PremiseResult(name=r["name"]) for r in results]
        return PremiseResults(items=items)
