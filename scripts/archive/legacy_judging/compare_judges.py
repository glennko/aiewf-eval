#!/usr/bin/env python3
"""
Compare judgments from OpenAI judge (alt) vs Claude Agent SDK judge.

This script analyzes disagreements between the two judges and provides
insights into which judge may be more accurate.

Usage:
    uv run compare_judges.py runs/20251119T051205
    uv run compare_judges.py runs/20251119T051205 --verbose
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def load_judgments(run_dir: Path, judge_type: str) -> List[Dict[str, Any]]:
    """Load judgments from either alt_judged.jsonl or claude_judged.jsonl."""
    if judge_type == "alt":
        path = run_dir / "alt_judged.jsonl"
    elif judge_type == "claude":
        path = run_dir / "claude_judged.jsonl"
    else:
        raise ValueError(f"Unknown judge type: {judge_type}")

    if not path.exists():
        return []

    records = []
    with path.open() as f:
        for line in f:
            records.append(json.loads(line))
    return records


def compare_judges(run_dir: Path, verbose: bool = False) -> None:
    """Compare judgments from both judges."""
    alt_judged = load_judgments(run_dir, "alt")
    claude_judged = load_judgments(run_dir, "claude")

    if not alt_judged:
        print(f"❌ No alt_judged.jsonl found in {run_dir}")
        print("   Run: uv run judge_transcript_alt.py", run_dir)
        return

    if not claude_judged:
        print(f"❌ No claude_judged.jsonl found in {run_dir}")
        print("   Run: uv run judge_transcript_claude.py", run_dir)
        return

    # Build dicts keyed by turn number
    alt_by_turn = {r["turn"]: r for r in alt_judged}
    claude_by_turn = {r["turn"]: r for r in claude_judged}

    # Find common turns
    common_turns = set(alt_by_turn.keys()) & set(claude_by_turn.keys())

    if not common_turns:
        print("❌ No common turns found between judges")
        return

    print(f"# Judge Comparison: {run_dir.name}")
    print(f"\nComparing {len(common_turns)} turns\n")

    # Aggregate metrics
    dimensions = ["tool_use_correct", "instruction_following", "kb_grounding"]
    agreements = {d: 0 for d in dimensions}
    disagreements = {d: [] for d in dimensions}
    alt_passes = {d: 0 for d in dimensions}
    claude_passes = {d: 0 for d in dimensions}

    for turn in sorted(common_turns):
        alt = alt_by_turn[turn]
        claude = claude_by_turn[turn]

        for dim in dimensions:
            alt_score = alt["scores"].get(dim, False)
            claude_score = claude["scores"].get(dim, False)

            if alt_score:
                alt_passes[dim] += 1
            if claude_score:
                claude_passes[dim] += 1

            if alt_score == claude_score:
                agreements[dim] += 1
            else:
                disagreements[dim].append({
                    "turn": turn,
                    "alt": alt_score,
                    "claude": claude_score,
                    "user": alt.get("user_text", ""),
                    "assistant": alt.get("assistant_text", ""),
                    "claude_reasoning": claude.get("claude_reasoning", ""),
                })

    # Print summary
    print("## Summary Metrics\n")
    print(f"| Dimension | Alt Passes | Claude Passes | Agreement | Disagreements |")
    print(f"|-----------|------------|---------------|-----------|---------------|")

    for dim in dimensions:
        total = len(common_turns)
        alt_pct = (alt_passes[dim] / total * 100) if total > 0 else 0
        claude_pct = (claude_passes[dim] / total * 100) if total > 0 else 0
        agree_pct = (agreements[dim] / total * 100) if total > 0 else 0
        disagree_count = len(disagreements[dim])

        print(
            f"| {dim.replace('_', ' ').title()} | "
            f"{alt_passes[dim]}/{total} ({alt_pct:.1f}%) | "
            f"{claude_passes[dim]}/{total} ({claude_pct:.1f}%) | "
            f"{agreements[dim]}/{total} ({agree_pct:.1f}%) | "
            f"{disagree_count} |"
        )

    # Overall agreement
    total_comparisons = len(common_turns) * len(dimensions)
    total_agreements = sum(agreements.values())
    overall_agreement_pct = (total_agreements / total_comparisons * 100) if total_comparisons > 0 else 0

    print(f"\n**Overall Agreement**: {total_agreements}/{total_comparisons} ({overall_agreement_pct:.1f}%)")

    # Print disagreements
    total_disagreements = sum(len(v) for v in disagreements.values())
    if total_disagreements > 0:
        print(f"\n## Disagreements ({total_disagreements} total)\n")

        for dim in dimensions:
            if disagreements[dim]:
                print(f"### {dim.replace('_', ' ').title()} ({len(disagreements[dim])} disagreements)\n")

                for d in disagreements[dim]:
                    print(f"**Turn {d['turn']}**")
                    print(f"- **User**: {d['user']}")
                    print(f"- **Assistant**: {d['assistant'][:200]}{'...' if len(d['assistant']) > 200 else ''}")
                    print(f"- **Alt Judge**: {'✅ Pass' if d['alt'] else '❌ Fail'}")
                    print(f"- **Claude Judge**: {'✅ Pass' if d['claude'] else '❌ Fail'}")
                    print(f"- **Claude's Reasoning**: {d['claude_reasoning']}")
                    print()

        # Analysis
        print("## Analysis\n")

        # Count who is stricter
        alt_stricter = sum(1 for dim in dimensions for d in disagreements[dim] if not d['alt'] and d['claude'])
        claude_stricter = sum(1 for dim in dimensions for d in disagreements[dim] if d['alt'] and not d['claude'])

        print(f"- **Alt Judge stricter**: {alt_stricter} cases")
        print(f"- **Claude Judge stricter**: {claude_stricter} cases")

        if alt_stricter > claude_stricter:
            print("\n**Conclusion**: Alt (OpenAI) judge is generally stricter.")
        elif claude_stricter > alt_stricter:
            print("\n**Conclusion**: Claude judge is generally stricter.")
        else:
            print("\n**Conclusion**: Both judges are equally strict.")

    else:
        print("\n✅ **Perfect Agreement** - Both judges agree on all dimensions for all turns!")

    # Verbose output
    if verbose and total_disagreements > 0:
        print("\n## Detailed Disagreement Analysis\n")

        # Group by turn
        turns_with_disagreements = set()
        for dim in dimensions:
            for d in disagreements[dim]:
                turns_with_disagreements.add(d["turn"])

        for turn in sorted(turns_with_disagreements):
            alt = alt_by_turn[turn]
            claude = claude_by_turn[turn]

            print(f"### Turn {turn}\n")
            print(f"**User**: {alt['user_text']}\n")
            print(f"**Assistant**: {alt['assistant_text']}\n")

            # Tool calls
            tool_calls = alt.get("tool_calls", [])
            if tool_calls:
                print(f"**Tool Calls**: {json.dumps(tool_calls, indent=2)}\n")

            # Scores comparison
            print("**Scores**:\n")
            print("| Dimension | Alt | Claude |")
            print("|-----------|-----|--------|")

            for dim in dimensions:
                alt_score = "✅" if alt["scores"].get(dim, False) else "❌"
                claude_score = "✅" if claude["scores"].get(dim, False) else "❌"
                marker = "⚠️" if alt_score != claude_score else ""
                print(f"| {dim} | {alt_score} | {claude_score} {marker} |")

            print(f"\n**Claude's Reasoning**: {claude.get('claude_reasoning', 'N/A')}\n")
            print("---\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare OpenAI judge vs Claude Agent SDK judge"
    )
    parser.add_argument(
        "run_dir",
        help="Path to runs/<timestamp> directory"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed disagreement analysis"
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: Run directory does not exist: {run_dir}")
        return

    compare_judges(run_dir, verbose=args.verbose)


if __name__ == "__main__":
    main()
