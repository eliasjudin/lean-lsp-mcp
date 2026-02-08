# lean-lsp-mcp

Remote-first Lean MCP server for OpenAI Responses/connector workflows.

## Breaking Changes

- Legacy `lean_*` tool names are removed.
- Supported transports are only `streamable-http` and `sse`.
- File tools require workspace-relative `path`.
- Single-tenant workspace per process (`LEAN_WORKSPACE_ROOT`).
- Tool exposure is split by profile (`read` vs `write`).

Migration details: [MIGRATION.md](MIGRATION.md)
Release notes: [CHANGELOG.md](CHANGELOG.md)

## Why this shape

This server follows OpenAI MCP guidance for connector/deep-research compatibility:

- `search`/`fetch` return one MCP `content` item of `type: "text"` with JSON payload ([MCP guide](https://platform.openai.com/docs/mcp#search-tool), [fetch](https://platform.openai.com/docs/mcp#fetch-tool)).
- Tool annotations communicate impact (`readOnlyHint`, `openWorldHint`, `destructiveHint`) ([Apps SDK MCP server guide](https://developers.openai.com/apps-sdk/build/mcp-server/#tool-annotations-and-elicitation)).
- Deployment docs include trust boundaries and approval controls for remote MCP servers ([Connectors and MCP servers](https://platform.openai.com/docs/guides/tools-connectors-mcp#approvals)).
- Tool input schemas reject unknown fields (`inputSchema.additionalProperties=false`) across the full tool surface.

## Install

### Users

Install from PyPI:

```bash
uv tool install lean-lsp-mcp
# or
pip install lean-lsp-mcp
```

### Developers

Clone this repo and install dev dependencies:

```bash
uv sync --extra dev
```

## Required configuration

```bash
export LEAN_WORKSPACE_ROOT=/absolute/path/to/lean/project
```

`LEAN_WORKSPACE_ROOT` must contain `lean-toolchain`.

## Authentication

Auth mode env:

- `LEAN_AUTH_MODE=oauth_and_bearer` (remote default)
- `LEAN_AUTH_MODE=oauth`
- `LEAN_AUTH_MODE=bearer`
- `LEAN_AUTH_MODE=mixed` (unauthenticated initialize/list-tools; per-tool OAuth/noauth via `securitySchemes`)
- `LEAN_AUTH_MODE=none` (only explicit local/test mode)

OAuth/bearer envs:

- `LEAN_OAUTH_ISSUER_URL`
- `LEAN_OAUTH_RESOURCE_SERVER_URL`
- `LEAN_OAUTH_REQUIRED_SCOPES` (comma-separated)
- `LEAN_LSP_MCP_TOKEN` (static bearer fallback)

Mixed/OAuth metadata endpoint:

- `GET /.well-known/oauth-protected-resource` returns resource metadata for OAuth discovery.
- In `mixed` mode, unauthenticated write calls return `_meta["mcp/www_authenticate"]` challenges.
- `list_tools` includes per-tool `securitySchemes` metadata (`noauth`/`oauth2`) to drive client auth behavior.

To allow unauthenticated local runs, set:

```bash
export LEAN_ALLOW_NO_AUTH=true
export LEAN_AUTH_MODE=none
```

## CLI flags and env overrides

The following CLI flags are currently supported and map to environment configuration:

| CLI flag | Behavior | Env mapping and related toggles |
| --- | --- | --- |
| `--auth-mode {none,oauth,bearer,oauth_and_bearer,mixed}` | Overrides authentication mode. | Sets `LEAN_AUTH_MODE`. Related: `LEAN_ALLOW_NO_AUTH`, `LEAN_OAUTH_ISSUER_URL`, `LEAN_OAUTH_RESOURCE_SERVER_URL`, `LEAN_OAUTH_REQUIRED_SCOPES`, `LEAN_LSP_MCP_TOKEN`. |
| `--loogle-local` | Enables local loogle backend (instead of remote API). | Sets `LEAN_LOOGLE_LOCAL=true`. Related: `LEAN_ENABLE_LOOGLE` to disable loogle tool entirely. |
| `--loogle-cache-dir <path>` | Overrides local loogle cache directory. | Sets `LEAN_LOOGLE_CACHE_DIR`. Default is `$XDG_CACHE_HOME/lean-lsp-mcp/loogle` or `~/.cache/lean-lsp-mcp/loogle`. |
| `--repl` | Enables REPL-based `multi_attempt` path. | Sets `LEAN_REPL=true`. Related: `LEAN_REPL_PATH` (binary override), `LEAN_REPL_MEM_MB` (memory cap). |
| `--repl-timeout <seconds>` | Sets REPL command timeout. | Sets `LEAN_REPL_TIMEOUT` (default: `60`). |

## Run the server

Read profile:

```bash
uv run python -m lean_lsp_mcp --transport streamable-http --profile read
# http://127.0.0.1:8000/mcp
```

Write profile:

```bash
uv run python -m lean_lsp_mcp --transport streamable-http --profile write
```

SSE transport:

```bash
uv run python -m lean_lsp_mcp --transport sse --profile read
# http://127.0.0.1:8000/sse
```

With auth/local-loogle/REPL overrides:

```bash
uv run python -m lean_lsp_mcp \
  --transport streamable-http \
  --profile write \
  --auth-mode mixed \
  --loogle-local \
  --loogle-cache-dir /tmp/lean-lsp-mcp-loogle \
  --repl \
  --repl-timeout 90
```

## Tool surface

Read profile tools:

- `search(query)`
- `fetch(id)`
- `outline(path)`
- `diagnostics(path, start_line?, end_line?, declaration_name?)`
- `goal(path, line, column?)`
- `term_goal(path, line, column?)`
- `hover(path, line, column)`
- `completions(path, line, column, max_completions?)`
- `declaration(path, symbol)`
- `local_search(query, limit?)`
- `leansearch(query, num_results?)`
- `loogle(query, num_results?)`
- `leanfinder(query, num_results?)`
- `state_search(path, line, column, num_results?)`
- `hammer_premise(path, line, column, num_results?)`
- `profile_proof(path, line, top_n?, timeout?)`

Write profile additional tools:

- `build(clean?, output_lines?)`
- `multi_attempt(path, line, snippets)`
- `run_code(code)`

## `search`/`fetch` MCP contract

`search` payload (`text` JSON):

```json
{
  "results": [
    {
      "id": "base64url-id",
      "title": "DeclarationName",
      "url": "https://docs.example.com/decl/base64url-id"
    }
  ]
}
```

`fetch` payload (`text` JSON):

```json
{
  "id": "base64url-id",
  "title": "DeclarationName",
  "text": "declaration source text",
  "url": "https://docs.example.com/decl/base64url-id",
  "metadata": {
    "path": "Mathlib/Algebra/...",
    "symbol": "DeclarationName"
  }
}
```

Set `LEAN_PUBLIC_BASE_URL` to a canonical HTTPS base used for citations.
If unset, the server falls back to `LEAN_OAUTH_RESOURCE_SERVER_URL`; for local-only workflows it can emit `lean://decl/...`.

## Remote trust boundaries

When exposing this server remotely:

- Treat MCP tool inputs/outputs as untrusted data crossing trust boundaries.
- Keep read and write profiles on separate endpoints/processes.
- Use explicit `allowed_tools` in Responses API calls.
- Keep approval enabled for sensitive actions unless you fully trust the endpoint.
- Validate/allowlist domains before using URLs returned by tool output ([URL safety note](https://platform.openai.com/docs/guides/tools-connectors-mcp#urls-within-mcp-tool-calls-and-outputs)).

## Responses API patterns

Read-only least privilege:

```json
{
  "model": "gpt-5",
  "input": "Find lemmas about commutativity and fetch the most relevant one.",
  "tools": [
    {
      "type": "mcp",
      "server_label": "lean-read",
      "server_url": "https://lean.example.com/mcp",
      "allowed_tools": ["search", "fetch", "hover", "goal"],
      "require_approval": {
        "never": { "tool_names": ["search", "fetch", "hover", "goal"] }
      }
    }
  ]
}
```

Write endpoint with approvals:

```json
{
  "model": "gpt-5",
  "input": "Run build and summarize errors.",
  "tools": [
    {
      "type": "mcp",
      "server_label": "lean-write",
      "server_url": "https://lean-write.example.com/mcp",
      "allowed_tools": ["build"],
      "require_approval": "always"
    }
  ]
}
```

Approval details: [Connectors and MCP servers - Approvals](https://platform.openai.com/docs/guides/tools-connectors-mcp#approvals)

## External provider toggles

External web-backed tools are enabled by default. Disable with:

- `LEAN_ENABLE_LEANSEARCH=false`
- `LEAN_ENABLE_LOOGLE=false`
- `LEAN_ENABLE_LEANFINDER=false`
- `LEAN_ENABLE_STATE_SEARCH=false`
- `LEAN_ENABLE_HAMMER_PREMISE=false`

## Development

Run lint + tests:

```bash
uv sync --extra dev
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run pytest tests/mcp/ -v
```
