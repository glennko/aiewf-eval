#!/usr/bin/env python3
"""Aggregate benchmark results across multiple runs per model.

This script:
1. Scans all run directories in runs/aiwf_medium_context/
2. Parses directory names (format: YYYYMMDDTHHMMSS_modelname) to extract model names
3. Loads claude_summary.json from each run containing judge scores
4. Groups runs by model and selects the 5 most recent for each
5. Aggregates scores (tool_use, instruction_following, kb_grounding) across runs
6. Loads TTFB values from transcript.jsonl for latency statistics
7. Calculates aggregate strict turn pass rate:
   turns where tool_use_correct && instruction_following && kb_grounding
8. Calculates median pass rate across individual runs (shows typical performance,
   less affected by outlier runs with model instability or duplicate tool calls)
9. Computes TTFB median, P95, and max across all turns
10. Outputs a ranked table sorted by aggregate pass rate
11. Saves detailed metadata to a timestamped JSON file
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import statistics

RUNS_DIR = Path("runs/aiwf_medium_context")
OUTPUT_DIR = Path("runs/aiwf_medium_context")

# Model name mappings - EXACT matches only
MODEL_MAPPINGS = {
    "gpt-4.1": "gpt-4.1",
    "gpt-5.1": "gpt-5.1",
    "gpt-5.2": "gpt-5.2",
    "gpt-5-mini": "gpt-5-mini",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "fiercefalcon": "fiercefalcon",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "claude-sonnet-4-5": "claude-sonnet-4-5",
    "claude-haiku-4-5": "claude-haiku-4-5",
    # Native audio models
    "gemini-2.5-flash-native-audio-preview-12-2025": "gemini-native-audio-12",
    "gemini-2.5-flash-native-audio-preview-09-2025": "gemini-native-audio-09",
    "gpt-realtime": "gpt-realtime",
    "amazon.nova-2-sonic-v1_0": "nova-sonic",
    "us.amazon.nova-2-pro-preview-20251202-v1_0": "nova-2-pro-preview",
}


def get_model_from_dir(dir_name: str) -> str | None:
    """Extract model name from directory name like 20251213T213018_gpt-4.1"""
    parts = dir_name.split("_", 1)
    if len(parts) == 2:
        model_part = parts[1]
        # Current run directories append an 8-char unique suffix.
        # Strip it so model matching stays stable.
        model_parts = model_part.rsplit("_", 1)
        if (
            len(model_parts) == 2
            and len(model_parts[1]) == 8
            and all(c in "0123456789abcdef" for c in model_parts[1].lower())
        ):
            model_part = model_parts[0]
        # EXACT matches only - no prefix matching
        if model_part in MODEL_MAPPINGS:
            return model_part
    return None


def load_summary(run_dir: Path) -> dict | None:
    """Load claude_summary.json from a run directory."""
    summary_file = run_dir / "claude_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)
    return None


def load_transcript_ttfb(run_dir: Path) -> list[float]:
    """Load TTFB values from transcript.jsonl."""
    transcript_file = run_dir / "transcript.jsonl"
    ttfb_values = []
    if transcript_file.exists():
        with open(transcript_file) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if "ttfb_ms" in record and record["ttfb_ms"] is not None:
                        ttfb_values.append(record["ttfb_ms"])
                except json.JSONDecodeError:
                    continue
    return ttfb_values


def load_turn_pass_from_judged(run_dir: Path) -> tuple[int, int]:
    """Count strict per-turn passes from claude_judged.jsonl."""
    judged_file = run_dir / "claude_judged.jsonl"
    if not judged_file.exists():
        return 0, 0

    passed = 0
    total = 0
    with judged_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            scores = rec.get("scores")
            if not isinstance(scores, dict):
                continue
            total += 1
            if (
                scores.get("tool_use_correct", False)
                and scores.get("instruction_following", False)
                and scores.get("kb_grounding", False)
            ):
                passed += 1

    return passed, total


def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark results")
    parser.add_argument("--model", "-m", help="Filter to specific model (e.g., gpt-4o)")
    parser.add_argument("--runs", "-r", type=int, default=5, help="Number of recent runs per model (default: 5)")
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    # Determine which models to include
    if args.model:
        if args.model not in MODEL_MAPPINGS:
            print(f"Error: Unknown model '{args.model}'. Available: {list(MODEL_MAPPINGS.keys())}")
            return
        models_to_include = {args.model}
        print(f"Filtering to model: {args.model}, using {args.runs} most recent runs")
    else:
        models_to_include = set(MODEL_MAPPINGS.keys())

    # Collect ALL runs by model first
    all_runs = defaultdict(list)

    # Scan all run directories
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue

        model = get_model_from_dir(run_dir.name)
        if model not in models_to_include:
            continue

        summary = load_summary(run_dir)
        if summary is None:
            continue

        all_runs[model].append({
            "dir": run_dir,
            "name": run_dir.name,
            "summary": summary,
        })

    # Now select only the N most recent runs per model (sorted by dir name = timestamp)
    results = defaultdict(lambda: {
        "tool_use": [],
        "instruction_following": [],
        "kb_grounding": [],
        "turn_pass": [],
        "turn_totals": [],
        "ttfb_values": [],
        "run_dirs": [],
        "per_run_stats": [],  # Individual run statistics
    })

    for model in models_to_include:
        runs = all_runs[model]
        # Sort by directory name (which includes timestamp) and take last N
        runs_sorted = sorted(runs, key=lambda x: x["name"])
        recent_runs = runs_sorted[-args.runs:]  # Take the N most recent

        for run in recent_runs:
            summary = run["summary"]
            passes = summary.get("claude_passes", {})
            turns_scored = int(summary.get("turns_scored", 0) or 0)
            tool = passes.get("tool_use_correct", 0)
            instr = passes.get("instruction_following", 0)
            kb = passes.get("kb_grounding", 0)
            turn_pass_count = 0
            turn_pass_total = 0
            turn_pass = summary.get("turn_pass")
            if isinstance(turn_pass, dict):
                turn_pass_count = int(turn_pass.get("count", 0) or 0)
                turn_pass_total = int(turn_pass.get("total", 0) or 0)
            if turn_pass_total <= 0:
                turn_pass_count, turn_pass_total = load_turn_pass_from_judged(run["dir"])
            if turn_pass_total <= 0 and turns_scored > 0:
                turn_pass_total = turns_scored
                turn_pass_count = min(tool, instr, kb)

            results[model]["tool_use"].append(tool)
            results[model]["instruction_following"].append(instr)
            results[model]["kb_grounding"].append(kb)
            results[model]["turn_pass"].append(turn_pass_count)
            results[model]["turn_totals"].append(turn_pass_total)
            results[model]["run_dirs"].append(run["name"])

            # Load TTFB values
            ttfb_values = load_transcript_ttfb(run["dir"])
            results[model]["ttfb_values"].extend(ttfb_values)

            # Store per-run stats
            ttfb_med = statistics.median(ttfb_values) if ttfb_values else 0
            results[model]["per_run_stats"].append({
                "run_dir": run["name"],
                "tool_use_correct": tool,
                "instruction_following": instr,
                "kb_grounding": kb,
                "turn_pass": turn_pass_count,
                "turns": turn_pass_total,
                "pass_rate": (turn_pass_count / turn_pass_total * 100) if turn_pass_total > 0 else 0,
                "ttfb_median_ms": round(ttfb_med, 1),
            })

    # Calculate aggregated metrics
    aggregated = []
    metadata = {
        "analysis_timestamp": run_timestamp,
        "analysis_time_iso": datetime.now().isoformat(),
        "runs_per_model": args.runs,
        "turns_per_run": 30,
        "total_turns_per_model": args.runs * 30,
        "models": {},
    }

    for model in models_to_include:
        data = results[model]
        if not data["tool_use"]:
            continue

        n_runs = len(data["tool_use"])
        total_turns = sum(data["turn_totals"])

        tool_sum = sum(data["tool_use"])
        if_sum = sum(data["instruction_following"])
        kb_sum = sum(data["kb_grounding"])
        turn_pass_sum = sum(data["turn_pass"])

        pass_rate = (turn_pass_sum / total_turns * 100) if total_turns > 0 else 0

        # Median pass rate across individual runs (shows typical performance)
        per_run_pass_rates = [s["pass_rate"] for s in data["per_run_stats"]]
        median_pass_rate = statistics.median(per_run_pass_rates) if per_run_pass_rates else 0

        # TTFB statistics
        ttfb = sorted(data["ttfb_values"])
        if ttfb:
            ttfb_med = statistics.median(ttfb)
            ttfb_p95_idx = int(len(ttfb) * 0.95)
            ttfb_p95 = ttfb[ttfb_p95_idx] if ttfb_p95_idx < len(ttfb) else ttfb[-1]
            ttfb_max = max(ttfb)
        else:
            ttfb_med = ttfb_p95 = ttfb_max = 0

        aggregated.append({
            "model": model,
            "turn_pass": f"{turn_pass_sum}/{total_turns}",
            "tool_use": f"{tool_sum}/{total_turns}",
            "instruction_following": f"{if_sum}/{total_turns}",
            "kb_grounding": f"{kb_sum}/{total_turns}",
            "pass_rate": pass_rate,
            "median_pass_rate": median_pass_rate,
            "ttfb_med": ttfb_med,
            "ttfb_p95": ttfb_p95,
            "ttfb_max": ttfb_max,
        })

        # Store in metadata
        metadata["models"][model] = {
            "aggregate": {
                "turn_pass": turn_pass_sum,
                "tool_use_correct": tool_sum,
                "instruction_following": if_sum,
                "kb_grounding": kb_sum,
                "total_turns": total_turns,
                "pass_rate": round(pass_rate, 2),
                "median_pass_rate": round(median_pass_rate, 2),
                "ttfb_median_ms": round(ttfb_med, 1),
                "ttfb_p95_ms": round(ttfb_p95, 1),
                "ttfb_max_ms": round(ttfb_max, 1),
            },
            "runs": data["per_run_stats"],
        }

    # Sort by pass rate descending
    aggregated.sort(key=lambda x: x["pass_rate"], reverse=True)

    # Output table (without Run Directory column)
    print("  | Model                | Turn Pass | Tool Use  | Instruction | KB Ground | Aggr Rate | Median Rate | TTFB Med | TTFB P95 | TTFB Max |")
    print("  |----------------------|-----------|-----------|-------------|-----------|-----------|-------------|----------|----------|----------|")

    for row in aggregated:
        model = row["model"]
        turn_pass = row["turn_pass"]
        tool = row["tool_use"]
        instr = row["instruction_following"]
        kb = row["kb_grounding"]
        pass_rate = f"{row['pass_rate']:.1f}%"
        median_rate = f"{row['median_pass_rate']:.1f}%"
        ttfb_med = f"{int(row['ttfb_med'])}ms"
        ttfb_p95 = f"{int(row['ttfb_p95'])}ms"
        ttfb_max = f"{int(row['ttfb_max'])}ms"

        print(f"  | {model:<20} | {turn_pass:<9} | {tool:<9} | {instr:<11} | {kb:<9} | {pass_rate:<9} | {median_rate:<11} | {ttfb_med:<8} | {ttfb_p95:<8} | {ttfb_max:<8} |")

    # Save metadata to timestamped file
    output_file = OUTPUT_DIR / f"analysis_{run_timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {output_file}")


if __name__ == "__main__":
    main()
