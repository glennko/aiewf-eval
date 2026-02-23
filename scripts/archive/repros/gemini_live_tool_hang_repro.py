"""Minimal Gemini Live tool-call hang repro with heavy frame logging.

Run examples:
  uv run scripts/gemini_live_tool_hang_repro.py --model models/gemini-2.5-flash-native-audio-preview-09-2025
  uv run scripts/gemini_live_tool_hang_repro.py --model models/gemini-2.5-flash-native-audio-preview-09-2025 --continuous-silence

This feeds a single prerecorded turn (turns_audio/turn_011.wav) and waits for
Gemini to finish the turn. If no TTSStoppedFrame/LLMFullResponseEndFrame/turnComplete
arrives, the pipeline will time out and the frame log will show the missing end
frames.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, UTC
from pathlib import Path

from loguru import logger
from dotenv import load_dotenv

from pipecat.frames.frames import Frame, CancelFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from scripts.gemini_live_tool_fix_service import GeminiLiveToolFixedService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.frames.frames import InputTextRawFrame
from pipecat.transports.base_transport import TransportParams

from scripts.paced_input_transport import PacedInputTransport
from scripts.tts_stopped_assistant_transcript import TTSStoppedAssistantTranscriptProcessor
from tools_schema import ToolsSchemaForTest
from system_instruction_short import system_instruction


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class FrameLogger(FrameProcessor):
    """Log every frame that passes this point to a file."""

    def __init__(self, log_path: Path):
        super().__init__()
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        ts = now_iso()
        line = f"{ts}\t{direction.name}\t{frame.__class__.__name__}"
        # Append rather than re-read the file to avoid churn.
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


async def noop_tool_handler(params: FunctionCallParams):
    """Return success immediately; no dedup to surface duplicate emissions."""
    logger.info(f"[TOOL] name={params.function_name} args={params.arguments}")
    if params.result_callback:
        await params.result_callback({"status": "success"})


def build_pipeline(model: str, continuous_silence: bool, idle_timeout: int):
    run_dir = Path("runs") / datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Input transport
    tparams = TransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=24000,
        audio_in_channels=1,
        audio_in_passthrough=True,
    )

    pit = PacedInputTransport(params=tparams, pre_roll_ms=100, continuous_silence=continuous_silence)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is required for Gemini Live repro")
    llm = GeminiLiveToolFixedService(
        api_key=api_key,
        model=model if model.startswith("models/") else f"models/{model}",
        system_instruction=system_instruction,
        tools=ToolsSchemaForTest,
    )
    llm.register_function(None, noop_tool_handler)

    frame_log = FrameLogger(run_dir / "frame_log.txt")
    assistant_shim = TTSStoppedAssistantTranscriptProcessor()

    pipeline = Pipeline(
        [
            pit,
            llm,
            frame_log,
            assistant_shim,
        ]
    )

    task = PipelineTask(
        pipeline,
        idle_timeout_secs=idle_timeout,
        idle_timeout_frames=(),
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    return run_dir, pit, task


async def enqueue_audio(pit: PacedInputTransport, audio_path: Path, delay: float = 1.0):
    await asyncio.sleep(delay)
    pit.enqueue_wav_file(str(audio_path))
    logger.info(f"Enqueued audio: {audio_path}")


async def enqueue_text(task: PipelineTask, text: str, delay: float = 0.2):
    from pipecat.frames.frames import LLMRunFrame  # local import to avoid lint noise

    await asyncio.sleep(delay)
    await task.queue_frames([InputTextRawFrame(text=text), LLMRunFrame()])
    logger.info(f"Enqueued text: {text[:120]}...")


async def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", default="turns_audio/turn_011.wav")
    parser.add_argument("--idle-timeout", type=int, default=30)
    parser.add_argument("--continuous-silence", action="store_true")
    parser.add_argument("--text-user", default="", help="Optional user text to send before/with audio")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    run_dir, pit, task = build_pipeline(
        model=args.model,
        continuous_silence=args.continuous_silence,
        idle_timeout=args.idle_timeout,
    )

    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Frame log: {run_dir/'frame_log.txt'}")

    # kick off audio
    asyncio.create_task(enqueue_audio(pit, audio_path))
    if args.text_user:
        asyncio.create_task(enqueue_text(task, args.text_user))

    runner = PipelineRunner(handle_sigint=True)
    await runner.run(task)

    # Always cancel at end to flush logs
    await task._pipeline.push_frame(CancelFrame())
    logger.info("Repro finished")


if __name__ == "__main__":
    asyncio.run(main())
from dotenv import load_dotenv
