#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_FILE="${PROJECT_ROOT}/evaluations/lean_lsp_readonly.xml"
REPORT_DIR="${PROJECT_ROOT}/reports"
REPORT_FILE="${REPORT_DIR}/lean_lsp_readonly.md"
METRICS_FILE="${REPORT_DIR}/evaluation_runs.jsonl"
SUMMARY_MD="${REPORT_DIR}/feedback_summary.md"
SUMMARY_JSON="${REPORT_DIR}/feedback_summary.json"

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ANTHROPIC_API_KEY is not set; cannot run evaluations." >&2
  exit 1
fi

mkdir -p "${REPORT_DIR}"

python "${PROJECT_ROOT}/mcp-builder/scripts/evaluation.py" \
  -t stdio \
  -c python \
  -a -m \
  -a lean_lsp_mcp \
  -a --transport \
  -a stdio \
  -e "LEAN_PROJECT_PATH=${PROJECT_ROOT}/evaluations/lean_playground" \
  -o "${REPORT_FILE}" \
  --metrics-json "${METRICS_FILE}" \
  "${EVAL_FILE}"

echo "Evaluation report written to ${REPORT_FILE}"

python "${PROJECT_ROOT}/scripts/analyze_evaluation_feedback.py" \
  --metrics-json "${METRICS_FILE}" \
  --output-md "${SUMMARY_MD}" \
  --output-json "${SUMMARY_JSON}"

echo "Feedback summary written to ${SUMMARY_MD}"
