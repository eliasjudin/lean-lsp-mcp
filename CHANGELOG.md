# Changelog

All notable changes to this project will be documented in this file.

## [0.20.1] - Unreleased

Release prep focused on CI hardening, test cleanup, and release process polish.

### CI hardening

- Run integration tests on `pull_request` and `push` to `main` (in addition to scheduled/manual runs).
- Remove fragile integration workflow steps (`lake update` / `lake exe cache get`) in favor of deterministic `lake build`.
- Provision Lean toolchain explicitly in CI workflows and cache toolchain/artifacts.
- Align release preflight checks with CI (`ruff check`, `ruff format --check`, `pytest tests/mcp/ -v`).
- Verify artifacts with `uv build` before creating or pushing the release tag.

### Test cleanup

- Tighten local test guidance around the `tests/mcp` suite used in CI.
- Keep developer setup and release checks consistent to reduce test drift.

### Release process polish

- Harden `release.sh` with `set -euo pipefail`.
- Replace GNU-specific `sed -i` usage with a Python-based version update.
- Remove automatic `uv publish`; publishing is now a separate explicit step.
