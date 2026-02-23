#!/usr/bin/env python3
"""
Standalone sanity check for the LLM-as-a-judge argument matcher on one turn.

Usage examples:
  python scripts/sanity_arg_match_llm.py runs/20250909T181605 --turn 17 --arg issue_description
  python scripts/sanity_arg_match_llm.py runs/20250909T181605 --turn 24 --arg session_id

This replicates arg_match_llm() from judge_transcript_alt.py for a single
expected-vs-provided argument pair, prints the exact system and user prompts
sent to the judge model, and shows the raw model response.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI


def norm_str(s: str) -> str:
    t = (s or "").lower().strip()
    t = t.replace("can't", "cannot").replace("cant", "cannot")
    t = t.replace("can not", "cannot")
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def load_expected(turn_idx: int) -> Dict[str, Any]:
    from turns import turns as expected_turns
    return expected_turns[turn_idx]


def load_actual(run_dir: Path, turn_idx: int) -> Dict[str, Any]:
    # Prefer alt_judged.jsonl since it has tool_calls flattened
    path = run_dir / "alt_judged.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("turn") == turn_idx:
                return rec
    raise RuntimeError(f"Turn {turn_idx} not found in {path}")


def build_system_prompt() -> str:
    return (
        "Decide if the PROVIDED_ARG conveys the SAME MEANING as the EXPECTED_ARG for the tool argument named ARG_NAME.\n"
        "Evaluate semantic equivalence, not verbatim match. Be liberal for short strings.\n"
        "Ignore: case, punctuation, articles (a, an, the), simple prepositions (in/on/at), contractions (can't=cannot), and word order.\n"
        "Accept near-synonyms and minor wording changes.\n"
        "Return strict JSON: {\"reasonable\": boolean}. Set reasonable=true if they express the same request/fact; set false only if they refer to different content or contradict.\n\n"
        "Examples:\n"
        "- TRUE: EXPECTED_ARG: \"Cannot access location maps on the mobile app.\" PROVIDED_ARG: \"Can't access the location maps in the mobile app\"\n"
        "- TRUE: EXPECTED_ARG: \"A session about open telemetry tracing.\" PROVIDED_ARG: \"OpenTelemetry tracing session\"\n"
        "- FALSE: EXPECTED_ARG: \"Cannot access location maps on the mobile app.\" PROVIDED_ARG: \"Need vegetarian lunch options\""
    )


def build_user_prompt(arg_name: str, expected: str, provided: str, user_text: str, assistant_text: str) -> str:
    return (
        f"ARG_NAME: {arg_name}\n"
        f"EXPECTED_ARG (reference): {expected}\n"
        f"PROVIDED_ARG: {provided}\n\n"
        f"USER_INPUT:\n{user_text}\n\nASSISTANT_ANSWER:\n{assistant_text}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--turn", type=int, required=True)
    ap.add_argument("--arg", dest="arg_name", required=True)
    ap.add_argument("--model", default=os.getenv("ALT_JUDGE_MODEL", "gpt-5"))
    args = ap.parse_args()

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    expected_turn = load_expected(args.turn)
    exp_fc = expected_turn.get("required_function_call") or {}
    exp_name = exp_fc.get("name")
    exp_args = (exp_fc.get("args") or {}).copy()
    if not exp_args:
        print(f"Turn {args.turn} has no required function call; comparing against empty-args is meaningless.")
        return 2
    if args.arg_name not in exp_args:
        print(f"Turn {args.turn} expected args: {list(exp_args)} (missing --arg {args.arg_name})")
        return 2

    rec = load_actual(args.run_dir, args.turn)
    user_text = rec.get("user_text", "")
    assistant_text = rec.get("assistant_text", "")
    # locate provided arg from first tool call matching exp_name
    provided = None
    for tc in (rec.get("tool_calls") or []):
        if tc.get("name") == exp_name:
            provided = (tc.get("args") or {}).get(args.arg_name)
            if provided is not None:
                break
    if provided is None:
        print("No matching tool call/arg in the actual record.")
        return 2

    expected = str(exp_args[args.arg_name])

    # Print local equivalence diagnostics
    print("=== Local diagnostics ===")
    print("Expected:", expected)
    print("Provided:", provided)
    print("norm equal?", norm_str(expected) == norm_str(provided))

    system = build_system_prompt()
    user_msg = build_user_prompt(args.arg_name, expected, provided, user_text, assistant_text)

    print("\n=== System Prompt ===\n" + system)
    print("\n=== User Message ===\n" + user_msg)

    print("\n=== Calling model ===")
    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=256,
        extra_body={"reasoning_effort": "low"},
        seed=41,
    )
    text = resp.choices[0].message.content or "{}"
    print("\n=== Raw model content ===\n" + text)
    try:
        data = json.loads(text)
    except Exception as e:  # noqa: BLE001
        print("JSON parse error:", e)
        return 3
    print("\n=== Parsed ===\n", data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

