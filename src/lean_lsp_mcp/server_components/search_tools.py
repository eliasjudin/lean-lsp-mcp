from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, List

from mcp.server.fastmcp import Context

from lean_lsp_mcp.schema_types import (
    ERROR_BAD_REQUEST,
    ERROR_NO_GOAL,
    ERROR_UNKNOWN,
)
from lean_lsp_mcp.tool_inputs import (
    LeanHammerPremiseInput,
    LeanSearchInput,
    LeanStateSearchInput,
    LoogleSearchInput,
)
from lean_lsp_mcp.tool_spec import TOOL_ANNOTATIONS
from lean_lsp_mcp.url_validator import validate_api_url
from lean_lsp_mcp.utils import format_line

from .common import (
    ToolError,
    _compact_pos,
    _set_response_format_hint,
    error_result,
    open_file_session,
    rate_limited,
    retrieve_goal_state_with_session,
    sanitize_exception,
    success_result,
)
from .context import mcp



def _build_service_url(base_url: str, endpoint: str) -> str:
    """Join a normalized service base URL with an endpoint path.

    Args:
        base_url: Normalized base URL without a trailing slash.
        endpoint: Endpoint path (leading slash optional).

    Returns:
        Fully qualified endpoint URL.
    """
    parsed = urllib.parse.urlparse(base_url)
    base_segments = [segment for segment in parsed.path.split("/") if segment]
    endpoint_segments = [segment for segment in endpoint.split("/") if segment]
    overlap = 0
    max_overlap = min(len(base_segments), len(endpoint_segments))
    for candidate in range(max_overlap, 0, -1):
        if base_segments[-candidate:] == endpoint_segments[:candidate]:
            overlap = candidate
            break
    path_segments = base_segments + endpoint_segments[overlap:]
    final_path = "/" + "/".join(path_segments) if path_segments else "/"
    rebuilt = parsed._replace(path=final_path, params="", query="", fragment="")
    return urllib.parse.urlunparse(rebuilt)


__all__ = ["leansearch", "loogle", "state_search", "hammer_premise"]


@mcp.tool(
    "lean_leansearch",
    description="Query leansearch.net for Lean lemmas/theorems; responses list fully qualified names in `structured.names` (3 req/30s rate limit).",
    annotations=TOOL_ANNOTATIONS["lean_leansearch"],
)
@rate_limited("leansearch", max_requests=3, per_seconds=30)
def leansearch(ctx: Context, params: LeanSearchInput) -> Any:
    started = time.perf_counter()
    query = params.query
    num_results = params.num_results
    response_format = params.response_format
    _set_response_format_hint(ctx, response_format)

    try:
        headers = {"User-Agent": "lean-lsp-mcp/0.1", "Content-Type": "application/json"}
        payload = json.dumps({"num_results": str(num_results), "query": [query]}).encode(
            "utf-8"
        )

        req = urllib.request.Request(
            "https://leansearch.net/search",
            data=payload,
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=20) as response:
            results = json.loads(response.read().decode("utf-8"))

        if not results or not results[0]:
            return error_result(
                message="No results found.",
                code=ERROR_UNKNOWN,
                category="lean_leansearch",
                details={"query": query, "num_results": num_results},
                hints=[
                    "Rephrase the query or add more specific search terms before retrying.",
                    "Alternative: Use lean_loogle for type-based search (e.g., '(?a -> ?b) -> List ?a -> List ?b').",
                    "Alternative: Use lean_state_search if you have a specific goal state to match.",
                ],
                start_time=started,
                ctx=ctx,
            )
        results = results[0][:num_results]
        results = [r["result"] for r in results]

        for result in results:
            result.pop("docstring", None)

            module_parts = result.get("module_name")
            if isinstance(module_parts, List):
                result["module_name"] = ".".join(module_parts)

            name_parts = result.get("name")
            if isinstance(name_parts, List):
                result["name"] = ".".join(name_parts)

        names = [str(res.get("name") or res.get("declaration") or "") for res in results]
        structured = {
            "query": query,
            "names": names,
            "pagination": {
                "count": len(results),
                "limit": num_results,
                "has_more": False,  # leansearch returns exact count requested
                "total_count": len(results),
            },
        }
        summary = f"{len(results)} results"
        return success_result(
            summary=summary,
            structured=structured,
            start_time=started,
            ctx=ctx,
        )
    except Exception as exc:
        error_message = sanitize_exception(exc, fallback_reason="leansearch.net API request")
        return error_result(
            message=f"leansearch.net request failed: {error_message}",
            code=ERROR_UNKNOWN,
            category="lean_leansearch",
            details={"query": query, "num_results": num_results},
            hints=[
                "Verify network connectivity to leansearch.net",
                "The service may be temporarily unavailable - try again in a moment.",
                "Alternative: Use lean_loogle for pattern-based search.",
                "Alternative: Use lean_state_search for goal-based theorem search.",
            ],
            start_time=started,
            ctx=ctx,
        )


@mcp.tool(
    "lean_loogle",
    description="Query loogle.lean-lang.org for Lean declarations by name/pattern; compact responses populate `structured.names` (3 req/30s).",
    annotations=TOOL_ANNOTATIONS["lean_loogle"],
)
@rate_limited("loogle", max_requests=3, per_seconds=30)
def loogle(ctx: Context, params: LoogleSearchInput) -> Any:
    started = time.perf_counter()
    query = params.query
    num_results = params.num_results
    response_format = params.response_format
    _set_response_format_hint(ctx, response_format)
    try:
        req = urllib.request.Request(
            f"https://loogle.lean-lang.org/json?q={urllib.parse.quote(query)}",
            headers={"User-Agent": "lean-lsp-mcp/0.1"},
            method="GET",
        )

        with urllib.request.urlopen(req, timeout=20) as response:
            results = json.loads(response.read().decode("utf-8"))

        if "hits" not in results:
            return error_result(
                message="No results found.",
                code=ERROR_UNKNOWN,
                category="lean_loogle",
                details={"query": query, "num_results": num_results},
                hints=[
                    "Ensure the query syntax is valid (e.g., quotes for substrings or `_` wildcards).",
                    "Alternative: Use lean_leansearch for natural language search.",
                    "Alternative: Use lean_state_search if you have a goal state.",
                ],
                start_time=started,
                ctx=ctx,
            )

        results = results["hits"][:num_results]
        if not results:
            return error_result(
                message="No results found.",
                code=ERROR_UNKNOWN,
                category="lean_loogle",
                details={"query": query, "num_results": num_results},
                hints=[
                    "Try a more general pattern or search by name substring (e.g., 'Real').",
                    "Alternative: Use lean_leansearch for natural language search.",
                    "Alternative: Use lean_state_search if you have a goal state.",
                ],
                start_time=started,
                ctx=ctx,
            )
        for result in results:
            result.pop("doc", None)
        names = [hit.get("name") for hit in results if hit.get("name")]
        structured = {
            "query": query,
            "names": names,
            "pagination": {
                "count": len(results),
                "limit": num_results,
                "has_more": False,  # loogle returns exact count requested
                "total_count": len(results),
            },
        }
        summary = f"{len(results)} results"
        return success_result(
            summary=summary,
            structured=structured,
            start_time=started,
            ctx=ctx,
        )
    except Exception as exc:
        error_message = sanitize_exception(exc, fallback_reason="loogle.lean-lang.org API request")
        return error_result(
            message=f"loogle.lean-lang.org request failed: {error_message}",
            code=ERROR_UNKNOWN,
            category="lean_loogle",
            details={"query": query, "num_results": num_results},
            hints=[
                "Verify network connectivity to loogle.lean-lang.org",
                "The service may be temporarily unavailable - try again shortly.",
                "Alternative: Use lean_leansearch for natural language search.",
                "Alternative: Use lean_state_search for goal-based search.",
            ],
            start_time=started,
            ctx=ctx,
        )


@mcp.tool(
    "lean_state_search",
    description="Fetch goal-based lemma suggestions from premise-search.com (3 req/30s).",
    annotations=TOOL_ANNOTATIONS["lean_state_search"],
)
@rate_limited("lean_state_search", max_requests=3, per_seconds=30)
def state_search(ctx: Context, params: LeanStateSearchInput) -> Any:
    started = time.perf_counter()
    file_path = params.file_path
    line = params.line
    column = params.column
    num_results = params.num_results
    response_format = params.response_format
    _set_response_format_hint(ctx, response_format)

    try:
        with open_file_session(
            ctx,
            file_path,
            started=started,
            response_format=response_format,
            line=line,
            column=column,
            category="lean_state_search",
        ) as file_session:
            identity = file_session.identity
            file_contents = file_session.load_content()
            goal_state = retrieve_goal_state_with_session(file_session, line, column)
    except ToolError as exc:
        return exc.payload

    f_line = format_line(file_contents, line, column)
    if not goal_state or not goal_state.get("goals"):
        return error_result(
            message="No goals found",
            code=ERROR_NO_GOAL,
            category="lean_state_search",
            details={
                "file": identity["relative_path"],
                "line": line,
                "column": column,
            },
            start_time=started,
            ctx=ctx,
        )

    mathlib_rev = os.getenv("LEAN_STATE_SEARCH_REV", "v4.22.0")
    goal_text = goal_state["goals"][0]

    # Validate and normalize the API URL
    raw_url = os.getenv("LEAN_STATE_SEARCH_URL", "https://premise-search.com")
    try:
        base_url = validate_api_url(raw_url, "LEAN_STATE_SEARCH_URL")
    except ValueError as exc:
        return error_result(
            message=f"Invalid LEAN_STATE_SEARCH_URL configuration: {exc}",
            code=ERROR_BAD_REQUEST,
            category="lean_state_search",
            details={
                "url": raw_url,
                "file": identity["relative_path"],
            },
            hints=[
                "Set LEAN_STATE_SEARCH_URL to a valid http or https URL.",
                "Example: LEAN_STATE_SEARCH_URL=https://premise-search.com",
            ],
            start_time=started,
            ctx=ctx,
        )

    try:
        query_encoded = urllib.parse.quote(goal_text)
        endpoint = _build_service_url(base_url, "api/search")
        url = (
            f"{endpoint}"
            f"?query={query_encoded}"
            f"&results={num_results}"
            f"&rev={mathlib_rev}"
        )

        headers = {"User-Agent": "lean-lsp-mcp/0.1"}
        req = urllib.request.Request(url, headers=headers, method="GET")

        with urllib.request.urlopen(req, timeout=20) as response:
            results = json.loads(response.read().decode("utf-8"))

        if not results:
            return error_result(
                message=f"No premise suggestions found for goal at {identity['relative_path']}:{line}:{column}",
                code=ERROR_UNKNOWN,
                category="lean_state_search",
                details={
                    "file": identity["relative_path"],
                    "line": line,
                    "column": column,
                    "goal": goal_text,
                    "limit": num_results,
                    "rev": mathlib_rev,
                },
                hints=[
                    "The goal state may be too specific or not well-represented in the premise database.",
                    "Strategies to find relevant theorems:",
                    "  1. Use lean_leansearch with natural language: describe what you're trying to prove",
                    "  2. Use lean_loogle with type patterns: search for theorems with similar signatures",
                    "  3. Simplify the goal or break it into smaller subgoals before searching",
                    f"Note: Using Mathlib revision {mathlib_rev}. Set LEAN_STATE_SEARCH_REV if your project uses a different version.",
                    "Consider self-hosting premise-search for better revision support: https://github.com/ruc-ai4math/LeanStateSearch",
                ],
                start_time=started,
                ctx=ctx,
            )

        names = [res.get("name") or "" for res in results]
        structured = {
            "file": {
                "uri": identity["uri"],
                "path": identity["relative_path"],
            },
            "pos": _compact_pos(line=line, column=column),
            "query": {
                "state": goal_text,
                "limit": num_results,
                "lineSnippet": f_line,
                "rev": mathlib_rev,
            },
            "names": names,
            "results": results,
            "pagination": {
                "count": len(results),
                "limit": num_results,
                "has_more": False,  # premise-search returns exact count requested
                "total_count": len(results),
            },
        }
        summary = f"{len(results)} results"
        return success_result(
            summary=summary,
            structured=structured,
            start_time=started,
            ctx=ctx,
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 405:
            return error_result(
                message="premise-search.com API method not allowed",
                code=ERROR_UNKNOWN,
                category="lean_state_search",
                details={
                    "error": "The API may have changed. Please report this issue.",
                    "http_code": 405,
                    "file": identity["relative_path"],
                },
                hints=[
                    "The premise-search.com API interface may have been updated.",
                    "Check https://premise-search.com for current API documentation.",
                    "Consider using a self-hosted instance via LEAN_STATE_SEARCH_URL.",
                ],
                start_time=started,
                ctx=ctx,
            )
        if exc.code == 400:
            try:
                error_data = json.loads(exc.read().decode("utf-8"))
                error_msg = error_data.get("error")
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                error_msg = None

            if not error_msg:
                error_msg = sanitize_exception(exc, fallback_reason="premise-search.com HTTP 400")

            is_revision_error = "revision" in error_msg.lower()

            return error_result(
                message=f"Premise search API error: {error_msg}",
                code=ERROR_BAD_REQUEST,
                category="lean_state_search",
                details={
                    "error": error_msg,
                    "http_code": 400,
                    "file": identity["relative_path"],
                    "rev": mathlib_rev,
                    "url": base_url,
                },
                hints=[
                    f"The Mathlib revision '{mathlib_rev}' may not be available on {base_url}.",
                    "The public premise-search.com instance may have limited revision support.",
                    "Recommended: Self-host for production use: https://github.com/ruc-ai4math/LeanStateSearch",
                    "Set LEAN_STATE_SEARCH_URL to your self-hosted instance URL.",
                    "Alternative: Use lean_leansearch or lean_loogle for theorem search.",
                ]
                if is_revision_error
                else [
                    "Check that your goal state is properly formatted.",
                    "Ensure the query parameter contains valid Lean syntax.",
                    f"API endpoint: {base_url}/api/search",
                ],
                start_time=started,
                ctx=ctx,
            )
        sanitized_error = sanitize_exception(exc, fallback_reason="HTTP error from premise-search.com")
        return error_result(
            message=f"premise-search.com HTTP error {exc.code}",
            code=ERROR_UNKNOWN,
            category="lean_state_search",
            details={
                "error": sanitized_error,
                "http_code": exc.code,
                "line": line,
                "column": column,
                "file": identity["relative_path"],
            },
            start_time=started,
            ctx=ctx,
        )
    except Exception as exc:
        sanitized_error = sanitize_exception(exc, fallback_reason="state search request")
        return error_result(
            message="state search error",
            code=ERROR_UNKNOWN,
            category="lean_state_search",
            details={
                "error": sanitized_error,
                "line": line,
                "column": column,
                "file": identity["relative_path"],
            },
            start_time=started,
            ctx=ctx,
        )


@mcp.tool(
    "lean_hammer_premise",
    description="Retrieve hammer premise suggestions for the active goal (3 req/30s).",
    annotations=TOOL_ANNOTATIONS["lean_hammer_premise"],
)
@rate_limited("hammer_premise", max_requests=3, per_seconds=30)
def hammer_premise(ctx: Context, params: LeanHammerPremiseInput) -> Any:
    started = time.perf_counter()
    file_path = params.file_path
    line = params.line
    column = params.column
    num_results = params.num_results
    response_format = params.response_format
    _set_response_format_hint(ctx, response_format)

    try:
        with open_file_session(
            ctx,
            file_path,
            started=started,
            response_format=response_format,
            line=line,
            column=column,
            category="lean_hammer_premise",
        ) as file_session:
            identity = file_session.identity
            file_contents = file_session.load_content()
            goal_state = retrieve_goal_state_with_session(file_session, line, column)
    except ToolError as exc:
        return exc.payload

    f_line = format_line(file_contents, line, column)
    if not goal_state or not goal_state.get("goals"):
        return error_result(
            message="No goals found",
            code=ERROR_NO_GOAL,
            category="lean_hammer_premise",
            details={
                "file": identity["relative_path"],
                "line": line,
                "column": column,
            },
            start_time=started,
            ctx=ctx,
        )

    data = {
        "state": goal_state["goals"][0],
        "new_premises": [],
        "k": num_results,
    }

    try:
        # Validate and normalize the API URL
        raw_url = os.getenv("LEAN_HAMMER_URL", "http://leanpremise.net")
        try:
            base_url = validate_api_url(raw_url, "LEAN_HAMMER_URL")
        except ValueError as exc:
            return error_result(
                message=f"Invalid LEAN_HAMMER_URL configuration: {exc}",
                code=ERROR_BAD_REQUEST,
                category="lean_hammer_premise",
                details={
                    "url": raw_url,
                    "file": identity["relative_path"],
                },
                hints=[
                    "Set LEAN_HAMMER_URL to a valid http or https URL.",
                    "Example: LEAN_HAMMER_URL=http://leanpremise.net",
                ],
                start_time=started,
                ctx=ctx,
            )
        
        endpoint = _build_service_url(base_url, "retrieve")
        req = urllib.request.Request(
            endpoint,
            headers={
                "User-Agent": "lean-lsp-mcp/0.1",
                "Content-Type": "application/json",
            },
            method="POST",
            data=json.dumps(data).encode("utf-8"),
        )

        with urllib.request.urlopen(req, timeout=20) as response:
            results = json.loads(response.read().decode("utf-8"))

        if not results:
            return error_result(
                message=f"No hammer premises found for goal at {identity['relative_path']}:{line}:{column}",
                code=ERROR_UNKNOWN,
                category="lean_hammer_premise",
                details={
                    "file": identity["relative_path"],
                    "line": line,
                    "column": column,
                    "goal": data["state"],
                    "limit": num_results,
                },
                hints=[
                    "The goal state may not have matching premises in the hammer database.",
                    "Strategies to find relevant theorems:",
                    "  1. Use lean_state_search for premise-search.com (different database)",
                    "  2. Use lean_leansearch with natural language description of your goal",
                    "  3. Use lean_loogle with type patterns matching your goal signature",
                    "  4. Simplify the goal or break it into smaller subgoals",
                    "Note: Consider self-hosting for better coverage: https://github.com/hanwenzhu/lean-premise-server",
                ],
                start_time=started,
                ctx=ctx,
            )

        premise_names = [result["name"] for result in results]
        structured = {
            "file": {
                "uri": identity["uri"],
                "path": identity["relative_path"],
            },
            "pos": _compact_pos(line=line, column=column),
            "query": {"state": data["state"], "limit": num_results, "lineSnippet": f_line},
            "names": premise_names,
            "pagination": {
                "count": len(results),
                "limit": num_results,
                "has_more": False,  # hammer returns exact count requested
                "total_count": len(results),
            },
        }
        summary = f"{len(results)} results"
        return success_result(
            summary=summary,
            structured=structured,
            start_time=started,
            ctx=ctx,
        )
    except Exception as exc:
        error_message = sanitize_exception(exc, fallback_reason="hammer premise API request")
        return error_result(
            message=f"Hammer premise search failed: {error_message}",
            code=ERROR_UNKNOWN,
            category="lean_hammer_premise",
            details={
                "file": identity["relative_path"],
                "line": line,
                "column": column,
                "limit": num_results,
            },
            hints=[
                "Verify the LEAN_HAMMER_URL is correctly configured (default: http://leanpremise.net)",
                "Check network connectivity to the hammer premise server.",
                "The service may be temporarily unavailable - try again shortly.",
                "Alternative: Use lean_state_search for premise-search.com",
                "Alternative: Use lean_leansearch for natural language search.",
            ],
            start_time=started,
            ctx=ctx,
        )
