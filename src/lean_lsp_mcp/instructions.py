INSTRUCTIONS = """## Workflow
- You are connected to the `lean_lsp_mcp` server; stay Lean-centric when reasoning about diagnostics and proofs.
- Begin by inspecting context with `lean_file_contents` or `lean_diagnostic_messages` before proposing edits.
- Use goal-oriented tools (`lean_goal`, `lean_term_goal`) repeatedly after each change to confirm progress.
- Reach for search helpers (`lean_leansearch`, `lean_loogle`, `lean_state_search`, `lean_hammer_premise`) only after local inspection narrows the need.
- `lean_build` runs `lake build` (optionally `lake clean`) and restarts the cached LSP client. Invoke it only when artifacts are stale.
- This MCP never writes files; coordinate with an external editor for permanent modifications.

## Tool Cheatsheet
- `lean_file_contents`: Read buffers with optional slices to view the current Lean source.
- `lean_diagnostic_messages`: List diagnostics around a file or position for triage.
- `lean_goal`: Inspect active goals; essential before applying new tactics.
- `lean_term_goal`: Check the expected term type when filling holes or definitions.
- `lean_hover_info`: Fetch type and doc info for a symbol at a given position.
- `lean_completions`: Request Lean completions for identifiers or tactics.
- `lean_declaration_file`: Open the file that defines a given declaration.
- `lean_multi_attempt`: Compare multiple tactic snippets against the current goal.
- `lean_run_code`: Execute an isolated Lean snippet (all imports required inline).
- `lean_tool_spec`: Export the current tool metadata for auditing.
- `lean_leansearch`: Query leansearch.net for relevant lemmas or theorems.
- `lean_loogle`: Search Loogle with name/type/sub-expression filters.
- `lean_state_search`: Goal-directed search that proposes matching lemmas.
- `lean_hammer_premise`: Retrieve hammer-suggested premises for the active goal.

## Rate Limits
- Remote discovery tools (`lean_leansearch`, `lean_loogle`, `lean_state_search`, `lean_hammer_premise`) are limited to 3 calls per 30 seconds each.
- If you hit a limit, pause or reformulate your query instead of retrying immediately; the server returns `ERROR_RATE_LIMIT` with timing details.

## Feedback-Driven Tips
- Diagnostics payloads expose `structured.summary.count` for quick totals and zero-based start indices in `structured.diags[*].s`; convert them to 1-based lines before relaying to users.
- When `lean_goal` reports no goals, ensure the cursor sits on non-whitespace or pass an explicit 1-based column to probe the target position.
- Hover responses package `structured.infoSnippet` plus nearby diagnostics—review both before escalating to external search.
- After running evaluations, inspect `reports/feedback_summary.md` to spot recurring tool usage issues and adjust strategies before the next run.

## Positioning and Safety
- Tool inputs use 1-indexed lines/columns. Raw LSP diagnostics remain 0-based—convert explicitly when correlating data.
- Always double-check the file identity (path + document version) before acting on goal output.
- Keep iterations small: explore, reason, then suggest the next Lean step.
"""
