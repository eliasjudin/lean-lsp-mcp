# Changelog

All notable changes to this project will be documented in this file.

## [0.21.0] - Unreleased

Breaking release that completes the MCP app migration on `app` and aligns the server with OpenAI MCP guidance.

### Breaking changes

- Remove legacy `lean_*` tool names.
- Keep only `streamable-http` and `sse` as supported transports.
- Enforce workspace-relative `path` handling for file tools.
- Split exposed tools by server profile (`read` / `write`) with no legacy compatibility aliases.

### OpenAI MCP conformance

- Keep `search`/`fetch` payloads aligned with MCP connector/deep-research contracts.
- Keep per-tool annotations and mixed-auth security metadata aligned with current guidance.
- Enforce strict input schemas (`additionalProperties: false`) across the full tool surface.
- Add MCP test coverage that validates strict schema behavior in both read and write profiles.

### CI and release

- Provision Lean toolchain and ripgrep in CI matrix jobs before running MCP tests.
- Run only the `tests/mcp` suite in CI after legacy test tree removal.
- Keep release preflight checks aligned with CI (`ruff`, `pytest tests/mcp`, `uv build`).
- Keep `release.sh` publishing manual and note fork-targeted PR flow.
