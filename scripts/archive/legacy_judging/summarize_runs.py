#!/usr/bin/env python3
"""
Summarize evaluation metrics from runs/*/alt_summary.json.

For each run directory under `runs/`, this script:
  1) Sanity checks that `alt_summary.json` exists and appears complete.
  2) Extracts the three metrics in `alt_passes` and prints a table row.

Completion heuristic:
  - File exists
  - Contains keys: model_name, alt_passes, turns_scored
  - alt_passes contains integer values for: tool_use_correct, instruction_following, kb_grounding

Usage:
  python scripts/summarize_runs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


RUNS_DIR = Path("runs")


def load_summary(run_dir: Path) -> Tuple[dict | None, str | None]:
    """Load and superficially validate alt_summary.json for a run.

    Returns (summary_dict, error_message). error_message is None on success.
    """
    summary_path = run_dir / "alt_summary.json"
    if not summary_path.exists():
        return None, "missing alt_summary.json"
    try:
        data = json.loads(summary_path.read_text())
    except Exception as e:  # noqa: BLE001
        return None, f"failed to parse alt_summary.json: {e}"

    # Required top-level keys
    for k in ("model_name", "alt_passes", "turns_scored"):
        if k not in data:
            return None, f"alt_summary.json missing key: {k}"

    passes = data.get("alt_passes", {})
    required_pass_keys = ("tool_use_correct", "instruction_following", "kb_grounding")
    for k in required_pass_keys:
        if k not in passes:
            return None, f"alt_passes missing key: {k}"
        v = passes[k]
        if not isinstance(v, int):
            return None, f"alt_passes[{k}] not int: {v!r}"

    return data, None


def format_table(rows: List[Dict[str, Any]]) -> str:
    # Determine column widths
    headers = [
        ("Model", "model_name"),
        ("ToolUse", "tool_use_correct"),
        ("InstrFollow", "instruction_following"),
        ("KBGround", "kb_grounding"),
        ("Turns", "turns_scored"),
        ("RunDir", "run_dir"),
    ]

    # Convert to strings early for width calc
    str_rows: List[List[str]] = []
    for r in rows:
        str_rows.append([
            str(r.get("model_name", "?")),
            str(r.get("tool_use_correct", "?")),
            str(r.get("instruction_following", "?")),
            str(r.get("kb_grounding", "?")),
            str(r.get("turns_scored", "?")),
            str(r.get("run_dir", "?")),
        ])

    widths = [len(h[0]) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(vals: List[str]) -> str:
        return "  ".join(val.ljust(widths[i]) for i, val in enumerate(vals))

    # Build table
    out_lines = []
    out_lines.append(fmt_row([h for h, _ in headers]))
    out_lines.append(fmt_row(["-" * w for w in widths]))
    for row in str_rows:
        out_lines.append(fmt_row(row))
    return "\n".join(out_lines)


def format_markdown(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "Model",
        "ToolUse",
        "InstrFollow",
        "KBGround",
        "Turns",
        "RunDir",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        vals = [
            str(r.get("model_name", "?")),
            str(r.get("tool_use_correct", "?")),
            str(r.get("instruction_following", "?")),
            str(r.get("kb_grounding", "?")),
            str(r.get("turns_scored", "?")),
            str(r.get("run_dir", "?")),
        ]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    if not RUNS_DIR.exists() or not RUNS_DIR.is_dir():
        print("runs/ directory not found.", file=sys.stderr)
        return 2

    rows: List[Dict[str, Any]] = []
    issues: List[str] = []

    for p in sorted(RUNS_DIR.iterdir()):
        if not p.is_dir():
            continue
        summary, err = load_summary(p)
        if err:
            issues.append(f"{p.name}: {err}")
            continue

        passes = summary["alt_passes"]
        rows.append(
            {
                "model_name": summary.get("model_name", ""),
                "tool_use_correct": passes.get("tool_use_correct"),
                "instruction_following": passes.get("instruction_following"),
                "kb_grounding": passes.get("kb_grounding"),
                "turns_scored": summary.get("turns_scored"),
                "run_dir": p.name,
            }
        )

    # Sort by model name for stable output
    rows.sort(key=lambda r: (str(r.get("model_name")), str(r.get("run_dir"))))

    if issues:
        print("Sanity check issues:")
        for msg in issues:
            print(f"  - {msg}")
        print()

    if not rows:
        print("No completed alt_summary.json files found under runs/.")
        return 1

    # Print text table to stdout
    print(format_table(rows))

    # If --markdown <path> provided, also write Markdown to that file
    md_out: str | None = None
    # Simple argv parse for --markdown PATH (avoid pulling argparse to keep script simple)
    import argparse as _argparse
    _p = _argparse.ArgumentParser(add_help=False)
    _p.add_argument("--markdown", dest="markdown", default=None)
    try:
        _ns, _ = _p.parse_known_args()
        md_out = _ns.markdown
    except SystemExit:
        md_out = None

    if md_out:
        out_path = Path(md_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(format_markdown(rows) + "\n", encoding="utf-8")
        print(f"\nWrote Markdown to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
