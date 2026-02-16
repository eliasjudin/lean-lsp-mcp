# Migration Guide: v1 -> current

## Scope

This release (`0.21.0`) is a hard break. Legacy `lean_*` tool names and legacy transports are removed.

## Tool Rename Map

| v1 tool | current tool |
|---|---|
| `lean_file_outline` | `outline` |
| `lean_diagnostic_messages` | `diagnostics` |
| `lean_goal` | `goal` |
| `lean_term_goal` | `term_goal` |
| `lean_hover_info` | `hover` |
| `lean_completions` | `completions` |
| `lean_declaration_file` | `declaration` |
| `lean_local_search` | `local_search` |
| `lean_leansearch` | `leansearch` |
| `lean_loogle` | `loogle` |
| `lean_leanfinder` | `leanfinder` |
| `lean_state_search` | `state_search` |
| `lean_hammer_premise` | `hammer_premise` |
| `lean_profile_proof` | `profile_proof` |
| `lean_build` | `build` |
| `lean_multi_attempt` | `multi_attempt` |
| `lean_run_code` | `run_code` |

New:
- `search(query)`
- `fetch(id)`

## Input/Output Contract Changes

- File-based inputs now use `path` (workspace-relative), not absolute `file_path`.
- Absolute paths and escape attempts (`..`, symlink escapes) are rejected.
- `search` now emits only workspace-fetchable declarations; each returned `id` is valid for `fetch` in the same workspace.
- `search` and `fetch` return explicit MCP tool results with:
  - exactly one `content` block of `type: "text"` containing JSON
  - `structuredContent` containing the same JSON object
- Shared JSON payloads:
  - `search`: `{"results":[{"id","title","url"}]}`
  - `fetch`: `{"id","title","text","url","metadata?"}`

## Transport Changes

Supported transports:
- `streamable-http` (default)
- `sse`
- `stdio` (local only — auth, CORS, and HTTP routes are disabled)

Launch with stdio for local editor integrations:
```bash
lean-lsp-mcp --transport stdio --workspace-root /path/to/project
```

HTTP-only flags (`--host`, `--port`, `--auth-mode`) are ignored in stdio mode and produce a warning if set.

## Authentication Changes

Auth mode is controlled by `LEAN_AUTH_MODE`:
- `oauth_and_bearer` (default for remote)
- `oauth`
- `bearer`
- `mixed` (initialize/list-tools unauthenticated; tools use per-tool `securitySchemes`)
- `none` (only for explicit local/test mode)

Related env vars:
- `LEAN_OAUTH_ISSUER_URL`
- `LEAN_OAUTH_RESOURCE_SERVER_URL`
- `LEAN_OAUTH_REQUIRED_SCOPES`
- `LEAN_LSP_MCP_TOKEN`

OAuth metadata route:
- `GET /.well-known/oauth-protected-resource` exposes resource metadata used by ChatGPT OAuth linking.
- In mixed mode, unauthorized write tools return `_meta["mcp/www_authenticate"]` challenges.

## Deployment Profile Changes

`LEAN_SERVER_PROFILE` controls exposed tools:
- `read`: read-only tool set
- `write`: read + write tools (`build`, `multi_attempt`, `run_code`)

MCP app/runtime envs:
- `LEAN_WORKSPACE_ROOT` is required and must point at a Lean project root.
- `LEAN_BIND_HOST` and `LEAN_BIND_PORT` control HTTP bind address (CLI: `--host`, `--port`).
- `LEAN_PUBLIC_BASE_URL` sets canonical citation URLs for `search`/`fetch`; fallback is `LEAN_OAUTH_RESOURCE_SERVER_URL`.

Recommended deployment:
1. Run a read endpoint for general model access.
2. Run a write endpoint with tighter access and approval policy.

## Build Behavior Change

- `build` no longer always runs `lake exe cache get`.
- To enable the old prefetch behavior explicitly, set `LEAN_BUILD_FETCH_CACHE=true`.
- Cache prefetch is best-effort and does not fail the tool by itself.

## Responses API Integration Notes

For least privilege, use MCP tool config with:
- `allowed_tools` to constrain callable tools.
- `require_approval` to enforce approval for sensitive/write actions.

For untrusted servers, keep approvals enabled and review data-sharing boundaries before allowing `never` approval.
