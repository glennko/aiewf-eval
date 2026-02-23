"""Lean Gemini Live conversation runner (text mode, tools enabled).

Usage:
    uv run scripts/convo-gemini-live-simple.py --model gemini-live-2.5-flash-preview-native-audio-09-2025
    uv run scripts/convo-gemini-live-simple.py --turn 11  # optional: pick a single turn

This bypasses paced audio, shims, and logging layers to isolate model/tool behavior.
"""

import asyncio
import os
import argparse
from pathlib import Path
from loguru import logger

from dotenv import load_dotenv
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response import LLMAssistantAggregatorParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.services.llm_service import FunctionCallParams

from scripts.gemini_live_tool_fix_service import GeminiLiveToolFixedService
from system_instruction_short import system_instruction
from tools_schema import ToolsSchemaForTest
from turns import turns

load_dotenv()


async def function_catchall(params: FunctionCallParams):
    logger.info(f"[FN] {params}")
    await params.result_callback({"status": "success"})


async def run(turn_idx: int, model: str):
    api_key = os.getenv("GOOGLE_API_KEY")
    assert api_key, "GOOGLE_API_KEY required"

    llm = GeminiLiveToolFixedService(
        api_key=api_key,
        model=model if model.startswith("models/") else f"models/{model}",
        system_instruction=system_instruction,
        tools=ToolsSchemaForTest,
    )
    llm.register_function(None, function_catchall)

    context = LLMContext([])
    agg = llm.create_context_aggregator(context, assistant_params=LLMAssistantAggregatorParams())

    pipeline = Pipeline(
        [
            agg.user(),
            llm,
            agg.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        idle_timeout_secs=120,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    user_text = turns[turn_idx]["input"]
    logger.info(f"[QUEUE] turn {turn_idx} text: {user_text}")
    await task.queue_frames(
        [
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": user_text}],
                run_llm=True,
            )
        ]
    )

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--turn", type=int, default=11, help="turn index to run")
    parser.add_argument(
        "--text",
        type=str,
        help="override user text (bypasses turns.py list) for quick repros",
    )
    args = parser.parse_args()

    if args.text is not None:
        # Quick inline run with custom text without touching turns.py
        from types import SimpleNamespace

        global turns
        turns = {args.turn: {"input": args.text}}

    asyncio.run(run(args.turn, args.model))


if __name__ == "__main__":
    main()
