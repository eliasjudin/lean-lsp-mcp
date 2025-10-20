#!/usr/bin/env python3
"""Summarise MCP evaluation metrics and surface recurring feedback."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:  # pragma: no cover - defensive import
    from lean_lsp_mcp.tool_spec import build_tool_spec
except Exception:  # pragma: no cover - fallback when import fails
    build_tool_spec = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate MCP evaluation metrics and produce feedback summaries.",
    )
    parser.add_argument(
        "--metrics-json",
        required=True,
        type=Path,
        help="Path to the JSONL file produced by evaluation.py --metrics-json.",
    )
    parser.add_argument(
        "--output-md",
        required=True,
        type=Path,
        help="Path to write the Markdown summary.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write aggregated metrics as JSON.",
    )
    return parser.parse_args()


def load_runs(metrics_path: Path) -> List[Dict[str, Any]]:
    if not metrics_path.exists():
        return []

    runs: List[Dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                runs.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skipping malformed JSON on line {line_number} of {metrics_path}: {exc}",
                    file=sys.stderr,
                )
    return runs


def percentile(values: List[float], quantile: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * quantile
    lower = int(k)
    upper = min(lower + 1, len(sorted_vals) - 1)
    weight = k - lower
    return sorted_vals[lower] + (sorted_vals[upper] - sorted_vals[lower]) * weight


def collect_tool_names(runs: List[Dict[str, Any]]) -> Set[str]:
    configured_names: Set[str] = set()
    if build_tool_spec is not None:
        try:
            configured_names = {tool["name"] for tool in build_tool_spec()["tools"]}
        except Exception:  # pragma: no cover - defensive guard
            configured_names = set()

    observed_names: Set[str] = set()
    for run in runs:
        for result in run.get("results", []):
            observed_names.update(result.get("tool_calls", {}).keys())

    return configured_names | observed_names


def aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    tool_names = collect_tool_names(runs)

    tool_usage: Dict[str, Dict[str, Any]] = {}
    feedback_hits: Dict[str, List[str]] = defaultdict(list)
    failed_tasks: List[Dict[str, Any]] = []

    total_tasks = 0
    total_missing_summary = 0
    total_missing_feedback = 0
    accuracies: List[float] = []

    for run_index, run in enumerate(runs):
        total_tasks += run.get("tasks", 0)
        total_missing_summary += run.get("missing_summary_count", 0)
        total_missing_feedback += run.get("missing_feedback_count", 0)

        accuracy = run.get("accuracy")
        if isinstance(accuracy, (int, float)):
            accuracies.append(float(accuracy))

        for result in run.get("results", []):
            if result.get("score", 0) == 0:
                failed_tasks.append(
                    {
                        "run": run_index,
                        "index": result.get("index"),
                        "question": result.get("question"),
                        "expected": result.get("expected"),
                        "actual": result.get("actual"),
                    }
                )

            for tool_name, metrics in result.get("tool_calls", {}).items():
                entry = tool_usage.setdefault(
                    tool_name,
                    {
                        "total_calls": 0,
                        "durations": [],
                        "tasks": 0,
                        "runs": set(),  # type: ignore[var-annotated]
                    },
                )
                count = metrics.get("count", 0)
                durations = metrics.get("durations") or []
                entry["total_calls"] += count
                entry["durations"].extend(durations)
                entry["tasks"] += 1
                entry["runs"].add(run_index)

            feedback_text = result.get("feedback") or ""
            if not feedback_text:
                continue

            lower_feedback = feedback_text.lower()
            seen_for_result: Set[str] = set()
            for tool_name in tool_names:
                tool_lower = tool_name.lower()
                if tool_lower and tool_lower in lower_feedback and tool_name not in seen_for_result:
                    feedback_hits[tool_name].append(feedback_text.strip())
                    seen_for_result.add(tool_name)

    return {
        "tool_usage": tool_usage,
        "feedback_hits": feedback_hits,
        "failed_tasks": failed_tasks,
        "total_tasks": total_tasks,
        "accuracies": accuracies,
        "missing_summary": total_missing_summary,
        "missing_feedback": total_missing_feedback,
    }


def format_markdown(
    runs: List[Dict[str, Any]],
    aggregates: Dict[str, Any],
) -> str:
    now_iso = datetime.now(timezone.utc).isoformat()
    run_count = len(runs)
    total_tasks = aggregates["total_tasks"]
    missing_summary = aggregates["missing_summary"]
    missing_feedback = aggregates["missing_feedback"]
    accuracies = aggregates["accuracies"]
    avg_accuracy = statistics.mean(accuracies) if accuracies else 0.0

    lines: List[str] = []
    lines.append("# MCP Evaluation Feedback Summary")
    lines.append("")
    lines.append(f"- Generated: {now_iso}")
    lines.append(f"- Runs analysed: {run_count}")
    lines.append(f"- Total tasks: {total_tasks}")
    lines.append(f"- Average accuracy: {avg_accuracy:.1f}%")
    lines.append(f"- Missing <summary> tags: {missing_summary}")
    lines.append(f"- Missing <feedback> tags: {missing_feedback}")
    lines.append("")

    lines.append("## Tool Usage Metrics")
    lines.append("")
    lines.append("| Tool | Runs Using | Total Calls | Avg Calls/Run | Avg Duration (s) | P95 Duration (s) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")

    tool_usage = aggregates["tool_usage"]
    if tool_usage:
        for tool_name in sorted(tool_usage.keys()):
            entry = tool_usage[tool_name]
            runs_with_tool = len(entry["runs"])
            total_calls = entry["total_calls"]
            avg_calls_per_run = (total_calls / run_count) if run_count else 0.0
            durations = entry["durations"]
            avg_duration = statistics.mean(durations) if durations else 0.0
            p95_duration = percentile(durations, 0.95) if durations else 0.0
            lines.append(
                f"| {tool_name} | {runs_with_tool} | {total_calls} | {avg_calls_per_run:.2f} | {avg_duration:.3f} | {p95_duration:.3f} |"
            )
    else:
        lines.append("| _No tool usage captured_ | 0 | 0 | 0 | 0 | 0 |")

    lines.append("")
    lines.append("## Feedback Highlights")
    lines.append("")

    feedback_hits: Dict[str, List[str]] = aggregates["feedback_hits"]
    if any(feedback_hits.values()):
        for tool_name, snippets in sorted(
            feedback_hits.items(),
            key=lambda item: len(item[1]),
            reverse=True,
        ):
            if not snippets:
                continue
            lines.append(f"- **{tool_name}** ({len(snippets)} mentions)")
            seen: Set[str] = set()
            for snippet in snippets:
                snippet_clean = " ".join(snippet.split())
                if snippet_clean in seen:
                    continue
                lines.append(f"  - {snippet_clean}")
                seen.add(snippet_clean)
                if len(seen) >= 3:
                    break
    else:
        lines.append("_No tool-specific feedback captured._")

    lines.append("")
    lines.append("## Incorrect Responses")
    lines.append("")
    failures = aggregates["failed_tasks"]
    if failures:
        for failure in failures[:10]:
            run_label = failure.get("run", "?")
            task_index = failure.get("index", "?")
            expected = failure.get("expected", "N/A")
            actual = failure.get("actual", "N/A")
            question = failure.get("question", "").strip()
            lines.append(
                f"- Run {run_label + 1 if isinstance(run_label, int) else run_label}, "
                f"Task {task_index + 1 if isinstance(task_index, int) else task_index}: "
                f"expected `{expected}`, got `{actual}`. Question: {question}"
            )
        if len(failures) > 10:
            lines.append(f"- … {len(failures) - 10} additional failures not shown.")
    else:
        lines.append("_All tasks answered correctly across analysed runs._")

    return "\n".join(lines) + "\n"


def build_json_summary(
    runs: List[Dict[str, Any]],
    aggregates: Dict[str, Any],
) -> Dict[str, Any]:
    tool_usage_summary = {
        tool_name: {
            "runs_using": len(entry["runs"]),
            "total_calls": entry["total_calls"],
            "avg_duration_s": statistics.mean(entry["durations"]) if entry["durations"] else 0.0,
            "p95_duration_s": percentile(entry["durations"], 0.95) if entry["durations"] else 0.0,
        }
        for tool_name, entry in aggregates["tool_usage"].items()
    }

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "runs_analysed": len(runs),
        "total_tasks": aggregates["total_tasks"],
        "average_accuracy": statistics.mean(aggregates["accuracies"]) if aggregates["accuracies"] else 0.0,
        "missing_summary_tags": aggregates["missing_summary"],
        "missing_feedback_tags": aggregates["missing_feedback"],
        "tool_usage": tool_usage_summary,
        "feedback_mentions": {tool: len(snippets) for tool, snippets in aggregates["feedback_hits"].items()},
        "failed_tasks": aggregates["failed_tasks"],
    }


def main() -> None:
    args = parse_args()
    runs = load_runs(args.metrics_json)

    if not runs:
        placeholder = (
            "# MCP Evaluation Feedback Summary\n\n"
            f"No evaluation runs found in `{args.metrics_json}`. "
            "Run evaluation.py with --metrics-json before analysing feedback.\n"
        )
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(placeholder, encoding="utf-8")
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps({}, indent=2), encoding="utf-8")
        print(f"Wrote placeholder feedback summary to {args.output_md}")
        return

    aggregates = aggregate_runs(runs)
    markdown = format_markdown(runs, aggregates)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(f"Saved feedback summary to {args.output_md}")

    if args.output_json:
        summary_json = build_json_summary(runs, aggregates)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
        print(f"Saved feedback metrics to {args.output_json}")


if __name__ == "__main__":
    main()
