#!/usr/bin/env python3
"""
Claude Agent SDK-based transcript judge (v3 with turn realignment).

This version adds intelligence to handle turn misalignment issues where:
- A function call happens earlier than expected (premature)
- Subsequent turns should not be penalized for the "missing" call

The judge uses a two-phase approach:
1. Initial pass: Compare each turn against golden expectations
2. Realignment pass: Detect early/late function calls and adjust scoring

Usage:
    uv run judge_transcript_claude_v3.py runs/aiwf_medium_context/20251215T202910_gemini-...
    uv run judge_transcript_claude_v3.py runs/... --only-turns 0,1,2
    uv run judge_transcript_claude_v3.py runs/... --debug
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv

try:
    from claude_agent_sdk import query, ClaudeAgentOptions
except ImportError:
    print("ERROR: claude-agent-sdk not installed.", file=sys.stderr)
    print("Install with: uv add claude-agent-sdk", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

JUDGE_VERSION = "claude-agent-sdk-v3-realign"
JUDGE_MODEL = "claude-opus-4-5"

# System prompt for the two-phase judge
JUDGE_SYSTEM_PROMPT = """# Role
You are an expert evaluator for conversational AI systems. You will judge a multi-turn conversation between a user and an AI assistant for the AI Engineer World's Fair 2025.

# Two-Phase Evaluation Process

You will evaluate in TWO phases:

## PHASE 1: Initial Turn-by-Turn Analysis
For each turn, evaluate against the golden expectation and note any discrepancies.

## PHASE 2: Realignment Analysis
After the initial pass, look for "turn misalignment" patterns:
- **Early function calls**: A function was called earlier than expected (e.g., at turn N instead of N+1)
- **Cascading effects**: If a function was called early, subsequent turns expecting that call should NOT be penalized
- **Semantic equivalence**: Even if timing differs, did the conversation accomplish the same goals?

# Evaluation Dimensions

For each turn, evaluate three dimensions:

1. **tool_use_correct** (bool):
   - TRUE if the assistant correctly called the expected function with semantically equivalent arguments
   - TRUE if no function call was expected and none was made
   - TRUE if a function call was expected but was already made in an earlier turn (realignment case)
   - FALSE if a function call was expected, not made, and NOT already made earlier
   - FALSE if the assistant's words imply waiting for confirmation but it acts without waiting
   - For argument matching, use semantic equivalence (not verbatim)
   - Session IDs must match exactly

2. **instruction_following** (bool):
   - TRUE if assistant directly answers the question OR advances the task
   - TRUE if assistant properly deflects out-of-scope questions
   - TRUE if the turn is part of a realigned workflow that still accomplishes the goal
   - FALSE if assistant's words contradict its actions (says "Does that work?" but doesn't wait)
   - FALSE if assistant neither answers nor advances the workflow

3. **kb_grounding** (bool):
   - TRUE unless assistant states an explicit factual error
   - TRUE if assistant provides additional correct information
   - FALSE only for clear factual contradictions (wrong dates, times, locations, speakers)

# Critical: Detecting Words-Actions Mismatch

A turn should FAIL instruction_following if the assistant's text implies one behavior but its actions show another:
- Says "I'll wait for confirmation" but calls the function immediately
- Says "Could you confirm?" but doesn't actually wait for the response
- Says "Does that work?" in the same turn where it confirms completion

# Critical: Handling Early Function Calls

When you detect an early function call:
1. Note which function was called and at which turn
2. In subsequent turns, if that same function was "expected", mark tool_use_correct as TRUE (already satisfied)
3. Add a note in reasoning explaining the realignment

# Output Format

Output a JSON object with this structure:
```json
{
  "phase1_analysis": [
    {"turn": 0, "initial_tool_use": true, "initial_instruction": true, "initial_kb": true, "notes": "..."},
    ...
  ],
  "realignment_notes": "Description of any detected misalignments and how they were resolved",
  "function_call_tracking": {
    "submit_dietary_request": {"expected_turn": 15, "actual_turn": 14, "status": "early"},
    ...
  },
  "final_judgments": [
    {"turn": 0, "reasoning": "...", "tool_use_correct": true, "instruction_following": true, "kb_grounding": true},
    ...
  ]
}
```

Output ONLY this JSON object, no markdown code blocks, no explanations outside the JSON.
"""


# ============================================================================
# Data Loading
# ============================================================================

def load_transcript(run_dir: Path) -> List[Dict[str, Any]]:
    """Load transcript.jsonl from run directory."""
    path = run_dir / "transcript.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No transcript.jsonl in {run_dir}")

    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ============================================================================
# Turn Formatting
# ============================================================================

def format_turns_for_claude(
    records: List[Dict[str, Any]],
    expected_turns: List[Dict[str, Any]],
    only_turns: Optional[set[int]] = None,
) -> str:
    """Format conversation turns with full context for realignment analysis."""
    lines = []

    # First, provide a summary of all expected function calls
    lines.append("# Expected Function Calls Summary")
    lines.append("")
    for i, exp in enumerate(expected_turns):
        fc = exp.get('required_function_call')
        if fc:
            lines.append(f"- Turn {i}: {fc['name']}({json.dumps(fc['args'])})")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Then provide each turn's details
    lines.append("# Conversation Turns")
    lines.append("")

    for rec in records:
        turn_idx = rec["turn"]

        # Skip turns not in the filter set
        if only_turns is not None and turn_idx not in only_turns:
            continue

        if turn_idx >= len(expected_turns):
            continue

        expected = expected_turns[turn_idx]

        lines.append(f"## Turn {turn_idx}")
        lines.append(f"**User**: {rec['user_text']}")
        lines.append(f"**Assistant**: {rec['assistant_text']}")
        lines.append("")

        golden = expected.get('golden_text', '')
        if golden:
            lines.append(f"**Golden Response**: {golden}")
            lines.append("")

        # Expected function call
        expected_fc = expected.get('required_function_call')
        if expected_fc:
            fc_str = json.dumps(expected_fc)
            lines.append(f"**Expected Function**: {fc_str}")
        else:
            lines.append("**Expected Function**: none")

        # Actual function calls
        actual_calls = rec.get('tool_calls', [])
        if actual_calls:
            calls_str = json.dumps(actual_calls)
            lines.append(f"**Actual Functions**: {calls_str}")
        else:
            lines.append("**Actual Functions**: none")

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# Claude Judge
# ============================================================================

async def judge_with_claude(
    run_dir: Path,
    only_turns: Optional[set[int]] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """Main judging function using two-phase realignment approach."""

    # Load data
    records = load_transcript(run_dir)
    from turns import turns as expected_turns

    # Filter records if only_turns specified
    if only_turns is not None:
        records = [r for r in records if r["turn"] in only_turns]

    if not records:
        raise ValueError("No turns to judge")

    model_name = records[0].get("model_name", "unknown")

    if debug:
        print(f"Judging {len(records)} turns with realignment analysis...", file=sys.stderr)

    # Format turns
    formatted_turns = format_turns_for_claude(records, expected_turns, only_turns)

    # Create prompt
    prompt = f"""{formatted_turns}

Please perform your two-phase evaluation:
1. First, analyze each turn against its golden expectation
2. Then, identify any turn misalignments (early/late function calls)
3. Apply realignment adjustments to avoid double-penalizing
4. Output the final JSON with all judgments

Remember:
- If a function is called early (before expected turn), subsequent turns should not be penalized for the "missing" call
- If the assistant says "Does that work?" but doesn't wait for confirmation, that's an instruction_following failure
- Be generous with kb_grounding unless there's a clear factual error
"""

    # Configure options - use extended thinking for complex reasoning
    options = ClaudeAgentOptions(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        model=JUDGE_MODEL,
        permission_mode="bypassPermissions",
    )

    # Query Claude
    all_text = []
    async for message in query(prompt=prompt, options=options):
        if hasattr(message, 'content'):
            if isinstance(message.content, str):
                all_text.append(message.content)
            elif isinstance(message.content, list):
                for block in message.content:
                    if hasattr(block, 'text'):
                        all_text.append(block.text)

    response_text = "".join(all_text)

    if debug:
        print(f"Claude response length: {len(response_text)} chars", file=sys.stderr)
        print(f"First 1000 chars:\n{response_text[:1000]}", file=sys.stderr)

    # Parse the JSON response
    # Try to find JSON object in the response
    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1

    if json_start == -1 or json_end == 0:
        raise ValueError(f"No JSON found in response: {response_text[:500]}")

    json_str = response_text[json_start:json_end]

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        if debug:
            print(f"JSON parse error: {e}", file=sys.stderr)
            print(f"Attempted to parse: {json_str[:500]}...", file=sys.stderr)
        raise ValueError(f"Failed to parse JSON response: {e}")

    # Extract final judgments
    final_judgments = result.get('final_judgments', [])
    realignment_notes = result.get('realignment_notes', '')
    function_tracking = result.get('function_call_tracking', {})

    if debug:
        print(f"\nRealignment notes: {realignment_notes}", file=sys.stderr)
        print(f"Function tracking: {json.dumps(function_tracking, indent=2)}", file=sys.stderr)

    # Convert to our standard format
    judgments = {}
    for j in final_judgments:
        turn_num = j.get('turn')
        if turn_num is not None:
            judgments[turn_num] = {
                "scores": {
                    "tool_use_correct": j.get('tool_use_correct', False),
                    "instruction_following": j.get('instruction_following', False),
                    "kb_grounding": j.get('kb_grounding', False),
                },
                "reasoning": j.get('reasoning', ''),
            }

    # Validate all turns were judged
    expected_turn_numbers = {r["turn"] for r in records}
    judged_turn_numbers = set(judgments.keys())
    missing = expected_turn_numbers - judged_turn_numbers

    if missing:
        raise ValueError(
            f"Failed to get judgments for turns: {sorted(missing)}. "
            f"Expected {len(expected_turn_numbers)} judgments, got {len(judgments)}."
        )

    return {
        "judgments": judgments,
        "realignment_notes": realignment_notes,
        "function_tracking": function_tracking,
        "summary": f"Evaluated {len(judgments)} turns with realignment.",
        "model_name": model_name,
    }


# ============================================================================
# Output Generation
# ============================================================================

def write_outputs(
    run_dir: Path,
    records: List[Dict[str, Any]],
    judgments: Dict[int, Dict[str, Any]],
    realignment_notes: str,
    function_tracking: Dict[str, Any],
    summary: str,
    model_name: str,
) -> None:
    """Write all output files."""

    # 1. claude_judged.jsonl
    with (run_dir / "claude_judged.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            turn = rec["turn"]
            judgment = judgments[turn]
            f.write(json.dumps({
                **rec,
                "scores": judgment["scores"],
                "claude_reasoning": judgment["reasoning"],
            }, ensure_ascii=False) + "\n")

    # 2. claude_summary.json
    passes = {
        "tool_use_correct": sum(
            1 for j in judgments.values() if j["scores"]["tool_use_correct"]
        ),
        "instruction_following": sum(
            1 for j in judgments.values() if j["scores"]["instruction_following"]
        ),
        "kb_grounding": sum(
            1 for j in judgments.values() if j["scores"]["kb_grounding"]
        ),
    }

    summary_data = {
        "model_name": model_name,
        "claude_passes": passes,
        "turns_scored": len(judgments),
        "judge_version": JUDGE_VERSION,
        "judge_model": JUDGE_MODEL,
        "judged_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "realignment_applied": bool(function_tracking),
        "function_tracking": function_tracking,
    }

    (run_dir / "claude_summary.json").write_text(
        json.dumps(summary_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    # 3. claude_analysis.md
    total = len(judgments)
    lines = [
        f"# Claude Agent SDK Evaluation (v3 with Realignment)",
        f"",
        f"**Model**: {model_name}",
        f"**Turns**: {total}",
        f"**Judge**: {JUDGE_MODEL}",
        f"**Judge Version**: {JUDGE_VERSION}",
        f"**Judged**: {summary_data['judged_at']}",
        f"",
        f"## Summary Metrics",
        f"",
        f"- **Tool Use Correct**: {passes['tool_use_correct']}/{total} ({passes['tool_use_correct']/total*100:.1f}%)",
        f"- **Instruction Following**: {passes['instruction_following']}/{total} ({passes['instruction_following']/total*100:.1f}%)",
        f"- **KB Grounding**: {passes['kb_grounding']}/{total} ({passes['kb_grounding']/total*100:.1f}%)",
        f"",
    ]

    # Add realignment notes if any
    if realignment_notes:
        lines.extend([
            f"## Realignment Analysis",
            f"",
            realignment_notes,
            f"",
        ])

    if function_tracking:
        lines.extend([
            f"## Function Call Tracking",
            f"",
            "| Function | Expected Turn | Actual Turn | Status |",
            "|----------|---------------|-------------|--------|",
        ])
        for func_name, tracking in function_tracking.items():
            exp = tracking.get('expected_turn', '?')
            act = tracking.get('actual_turn', '?')
            status = tracking.get('status', '?')
            lines.append(f"| {func_name} | {exp} | {act} | {status} |")
        lines.append("")

    lines.extend([
        f"## Per-Turn Failures",
        f"",
    ])

    # Add failure details
    has_failures = False
    for rec in records:
        turn = rec["turn"]
        judgment = judgments[turn]
        scores = judgment["scores"]

        if not all(scores.values()):
            has_failures = True
            failed_dimensions = [k for k, v in scores.items() if not v]

            lines.append(f"### Turn {turn}")
            lines.append(f"")
            lines.append(f"**User**: {rec['user_text']}")
            lines.append(f"")
            lines.append(f"**Assistant**: {rec['assistant_text'][:300]}{'...' if len(rec['assistant_text']) > 300 else ''}")
            lines.append(f"")
            lines.append(f"**Failed Dimensions**: {', '.join(failed_dimensions)}")
            lines.append(f"")
            lines.append(f"**Claude's Reasoning**: {judgment['reasoning']}")
            lines.append(f"")

    if not has_failures:
        lines.append("*No failures - all turns passed all evaluation dimensions!*")

    (run_dir / "claude_analysis.md").write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Judge conversation transcripts using Claude Agent SDK (v3 with realignment)"
    )
    parser.add_argument(
        "run_dir",
        help="Path to runs/<timestamp> directory containing transcript.jsonl"
    )
    parser.add_argument(
        "--only-turns",
        default="",
        help="Comma-separated list of turn indices to judge (e.g., '0,1,2')"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Validate ANTHROPIC_API_KEY
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
        print("Set it with: export ANTHROPIC_API_KEY=your_key_here", file=sys.stderr)
        sys.exit(1)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: Run directory does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    # Parse only_turns filter
    only_turns: Optional[set[int]] = None
    if args.only_turns.strip():
        try:
            only_turns = {int(x.strip()) for x in args.only_turns.split(',') if x.strip()}
            if args.debug:
                print(f"Filtering to turns: {sorted(only_turns)}", file=sys.stderr)
        except ValueError as e:
            print(f"ERROR: Invalid --only-turns format: {e}", file=sys.stderr)
            sys.exit(1)

    # Load records (for output generation)
    records = load_transcript(run_dir)
    if only_turns is not None:
        records = [r for r in records if r["turn"] in only_turns]

    # Run judgment
    try:
        result = asyncio.run(judge_with_claude(run_dir, only_turns, args.debug))
    except Exception as e:
        print(f"ERROR: Judgment failed: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Write outputs
    write_outputs(
        run_dir,
        records,
        result["judgments"],
        result.get("realignment_notes", ""),
        result.get("function_tracking", {}),
        result["summary"],
        result["model_name"],
    )

    # Print summary
    total = len(result["judgments"])
    passes = {
        "tool_use": sum(1 for j in result["judgments"].values() if j["scores"]["tool_use_correct"]),
        "instruction": sum(1 for j in result["judgments"].values() if j["scores"]["instruction_following"]),
        "kb": sum(1 for j in result["judgments"].values() if j["scores"]["kb_grounding"]),
    }

    print(f"Judged {total} turns (with realignment)")
    print(f"  Tool use: {passes['tool_use']}/{total}")
    print(f"  Instruction following: {passes['instruction']}/{total}")
    print(f"  KB grounding: {passes['kb']}/{total}")

    if result.get("realignment_notes"):
        print(f"\nRealignment applied: {result['realignment_notes'][:200]}...")

    if args.debug:
        print(f"\n✓ Wrote outputs:", file=sys.stderr)
        print(f"  - {run_dir / 'claude_judged.jsonl'}", file=sys.stderr)
        print(f"  - {run_dir / 'claude_summary.json'}", file=sys.stderr)
        print(f"  - {run_dir / 'claude_analysis.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
