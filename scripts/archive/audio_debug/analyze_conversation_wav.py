#!/usr/bin/env python3
"""Analyze conversation.wav for turn segmentation and timing verification.

This script:
1. Loads a stereo WAV file (user=left, bot=right)
2. Detects speech segments on each channel using energy-based VAD
3. Parses run.log for audio timing events
4. Compares audio segments with log timing
5. Reports any mismatches

Usage:
    python scripts/analyze_conversation_wav.py runs/aiwf_medium_context/<timestamp>_<model>/
"""

import argparse
import re
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Segment:
    """A speech segment with start/end times in milliseconds."""
    start_ms: float
    end_ms: float
    peak_db: float = 0.0

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass
class LogEvent:
    """An audio event from the log file."""
    timestamp: datetime
    event_type: str  # "SENDING", "FINISHED", "NullAudioOutput", etc.
    details: str


def load_stereo_wav(path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load stereo WAV file, return (user_audio, bot_audio, sample_rate)."""
    with wave.open(str(path), 'rb') as wf:
        n_channels = wf.getnchannels()
        if n_channels != 2:
            print(f"Warning: Expected stereo, got {n_channels} channels")
            if n_channels == 1:
                # Mono - treat as both user and bot
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
                audio = np.frombuffer(raw, dtype=np.int16)
                return audio.copy(), audio.copy(), sample_rate

        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        audio = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
        return audio[:, 0].copy(), audio[:, 1].copy(), sample_rate


def compute_energy_db(audio: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """Compute energy in dB for each frame."""
    n_samples = len(audio)
    n_frames = max(1, (n_samples - frame_size) // hop_size + 1)

    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_size
        end = min(start + frame_size, n_samples)
        frame = audio[start:end].astype(np.float64)
        rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
        energy[i] = 20 * np.log10(rms / 32768.0 + 1e-10)

    return energy


def find_speech_segments(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40,
    min_duration_ms: float = 100,
    merge_gap_ms: float = 200
) -> List[Segment]:
    """Find speech segments using energy-based detection.

    Args:
        audio: Audio samples (int16)
        sample_rate: Sample rate in Hz
        threshold_db: Energy threshold in dB (below this is silence)
        min_duration_ms: Minimum segment duration to keep
        merge_gap_ms: Merge segments closer than this

    Returns:
        List of Segment objects
    """
    frame_size = int(sample_rate * 0.025)  # 25ms frames
    hop_size = int(sample_rate * 0.010)    # 10ms hop

    energy = compute_energy_db(audio, frame_size, hop_size)

    # Find frames above threshold
    is_speech = energy > threshold_db

    # Find segment boundaries
    segments = []
    in_segment = False
    start_frame = 0

    for i, speech in enumerate(is_speech):
        if speech and not in_segment:
            # Start of segment
            in_segment = True
            start_frame = i
        elif not speech and in_segment:
            # End of segment
            in_segment = False
            start_ms = (start_frame * hop_size) / sample_rate * 1000
            end_ms = (i * hop_size) / sample_rate * 1000
            peak_db = float(np.max(energy[start_frame:i]))
            segments.append(Segment(start_ms=start_ms, end_ms=end_ms, peak_db=peak_db))

    # Handle segment that extends to end
    if in_segment:
        start_ms = (start_frame * hop_size) / sample_rate * 1000
        end_ms = len(audio) / sample_rate * 1000
        peak_db = float(np.max(energy[start_frame:]))
        segments.append(Segment(start_ms=start_ms, end_ms=end_ms, peak_db=peak_db))

    # Merge close segments
    merged = []
    for seg in segments:
        if merged and (seg.start_ms - merged[-1].end_ms) < merge_gap_ms:
            # Merge with previous
            merged[-1] = Segment(
                start_ms=merged[-1].start_ms,
                end_ms=seg.end_ms,
                peak_db=max(merged[-1].peak_db, seg.peak_db)
            )
        else:
            merged.append(seg)

    # Filter by minimum duration
    filtered = [s for s in merged if s.duration_ms >= min_duration_ms]

    return filtered


def parse_log_timing(log_path: Path) -> List[LogEvent]:
    """Parse run.log for audio timing events."""
    events = []

    # Patterns to match
    patterns = [
        (r"SENDING REAL AUDIO", "SENDING"),
        (r"FINISHED SENDING AUDIO", "FINISHED"),
        (r"\[NullAudioOutput\] First audio frame", "BOT_START"),
        (r"\[NullAudioOutput\] Frame (\d+):", "BOT_PROGRESS"),
        (r"BotStoppedSpeakingFrame", "BOT_STOP"),
        (r"\[TurnGate\] Triggering turn end", "TURN_END"),
        (r"Starting turn (\d+):", "TURN_START"),
    ]

    # Timestamp pattern: 2025-12-15 20:55:55.123 or similar
    timestamp_pattern = r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)"

    if not log_path.exists():
        print(f"Warning: Log file not found: {log_path}")
        return events

    with open(log_path) as f:
        for line in f:
            for pattern, event_type in patterns:
                if re.search(pattern, line):
                    # Try to extract timestamp
                    ts_match = re.search(timestamp_pattern, line)
                    if ts_match:
                        try:
                            ts_str = ts_match.group(1)
                            # Handle both with and without milliseconds
                            if '.' in ts_str:
                                timestamp = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                            else:
                                timestamp = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            timestamp = None
                    else:
                        timestamp = None

                    events.append(LogEvent(
                        timestamp=timestamp,
                        event_type=event_type,
                        details=line.strip()[:200]
                    ))
                    break

    return events


def format_ms(ms: float) -> str:
    """Format milliseconds as mm:ss.mmm."""
    total_secs = ms / 1000
    mins = int(total_secs // 60)
    secs = total_secs % 60
    return f"{mins:02d}:{secs:06.3f}"


def correlate_log_events_with_audio(
    log_events: List[LogEvent],
    user_segments: List[Segment],
    bot_segments: List[Segment],
    tolerance_ms: float = 500
) -> List[dict]:
    """Correlate log events with audio segments to detect timing mismatches.

    This function attempts to:
    1. Find the first log timestamp (recording start)
    2. Calculate relative times for all events
    3. Match user segments with SENDING/FINISHED events
    4. Match bot segments with BOT_START/BOT_STOP events
    5. Report timing differences

    Args:
        log_events: Parsed log events with timestamps
        user_segments: Detected user speech segments
        bot_segments: Detected bot speech segments
        tolerance_ms: Maximum acceptable timing difference

    Returns:
        List of timing analysis results
    """
    results = []

    # Find events with timestamps
    timestamped_events = [e for e in log_events if e.timestamp]
    if not timestamped_events:
        return results

    # Use first SENDING or BOT_START as reference point
    reference_event = None
    reference_segment = None
    for ev in timestamped_events:
        if ev.event_type == "SENDING" and user_segments:
            reference_event = ev
            reference_segment = user_segments[0]
            break
        elif ev.event_type == "BOT_START" and bot_segments:
            reference_event = ev
            reference_segment = bot_segments[0]
            break

    if not reference_event:
        return results

    # Calculate time offset (log time to audio time)
    log_reference_ms = 0  # First event is our zero point
    audio_reference_ms = reference_segment.start_ms

    # Group events by turn
    turn_events = []
    current_turn = {"user_start": None, "user_end": None, "bot_start": None, "bot_end": None}

    reference_ts = reference_event.timestamp
    for ev in timestamped_events:
        ev_offset_ms = (ev.timestamp - reference_ts).total_seconds() * 1000

        if ev.event_type == "SENDING":
            if current_turn["user_start"] is not None:
                turn_events.append(current_turn)
                current_turn = {"user_start": None, "user_end": None, "bot_start": None, "bot_end": None}
            current_turn["user_start"] = ev_offset_ms
        elif ev.event_type == "FINISHED":
            current_turn["user_end"] = ev_offset_ms
        elif ev.event_type == "BOT_START":
            current_turn["bot_start"] = ev_offset_ms
        elif ev.event_type == "BOT_STOP":
            current_turn["bot_end"] = ev_offset_ms

    if any(v is not None for v in current_turn.values()):
        turn_events.append(current_turn)

    # Match turns with segments
    for i, (turn, user_seg, bot_seg) in enumerate(zip(
        turn_events,
        user_segments[:len(turn_events)],
        bot_segments[:len(turn_events)]
    )):
        turn_result = {"turn": i + 1, "mismatches": []}

        # Adjust log times to audio reference
        log_offset = audio_reference_ms - log_reference_ms

        if turn["user_start"] is not None:
            expected_user_start = turn["user_start"] + log_offset
            actual_user_start = user_seg.start_ms
            delta = abs(expected_user_start - actual_user_start)
            turn_result["user_start_delta_ms"] = delta
            if delta > tolerance_ms:
                turn_result["mismatches"].append(
                    f"User start: expected {format_ms(expected_user_start)}, "
                    f"got {format_ms(actual_user_start)} (delta={delta:.0f}ms)"
                )

        if turn["bot_start"] is not None:
            expected_bot_start = turn["bot_start"] + log_offset
            actual_bot_start = bot_seg.start_ms
            delta = abs(expected_bot_start - actual_bot_start)
            turn_result["bot_start_delta_ms"] = delta
            if delta > tolerance_ms:
                turn_result["mismatches"].append(
                    f"Bot start: expected {format_ms(expected_bot_start)}, "
                    f"got {format_ms(actual_bot_start)} (delta={delta:.0f}ms)"
                )

        results.append(turn_result)

    return results


def analyze_run(run_dir: Path, verbose: bool = False, tolerance_ms: float = 500):
    """Analyze a run directory."""
    wav_path = run_dir / "conversation.wav"
    log_path = run_dir / "run.log"

    print(f"Analyzing: {run_dir}")
    print("=" * 60)

    # Check for required files
    if not wav_path.exists():
        print(f"ERROR: conversation.wav not found in {run_dir}")
        print("       (Audio recording may not be enabled for this pipeline)")
        return

    # Load audio
    print(f"\nLoading {wav_path}...")
    user_audio, bot_audio, sample_rate = load_stereo_wav(wav_path)

    duration_secs = len(user_audio) / sample_rate
    print(f"Duration: {duration_secs:.2f}s ({format_ms(duration_secs * 1000)})")
    print(f"Sample rate: {sample_rate}Hz")
    print(f"Samples: {len(user_audio):,} per channel")

    # Compute overall levels
    user_rms = 20 * np.log10(np.sqrt(np.mean(user_audio.astype(np.float64) ** 2)) / 32768.0 + 1e-10)
    bot_rms = 20 * np.log10(np.sqrt(np.mean(bot_audio.astype(np.float64) ** 2)) / 32768.0 + 1e-10)
    print(f"User channel RMS: {user_rms:.1f} dB")
    print(f"Bot channel RMS: {bot_rms:.1f} dB")

    # Find speech segments
    print("\n--- User Speech Segments ---")
    user_segments = find_speech_segments(user_audio, sample_rate)
    if not user_segments:
        print("  No speech detected (channel may be silent or threshold too high)")
    else:
        for i, seg in enumerate(user_segments):
            print(f"  {i+1}: {format_ms(seg.start_ms)} - {format_ms(seg.end_ms)} "
                  f"({seg.duration_ms:.0f}ms, peak={seg.peak_db:.1f}dB)")

    print("\n--- Bot Speech Segments ---")
    bot_segments = find_speech_segments(bot_audio, sample_rate)
    if not bot_segments:
        print("  No speech detected (channel may be silent or threshold too high)")
    else:
        for i, seg in enumerate(bot_segments):
            print(f"  {i+1}: {format_ms(seg.start_ms)} - {format_ms(seg.end_ms)} "
                  f"({seg.duration_ms:.0f}ms, peak={seg.peak_db:.1f}dB)")

    # Parse log events
    print("\n--- Log Events ---")
    log_events = parse_log_timing(log_path)

    if not log_events:
        print("  No relevant events found in log")
    else:
        # Group by type
        event_counts = {}
        for ev in log_events:
            event_counts[ev.event_type] = event_counts.get(ev.event_type, 0) + 1

        for event_type, count in sorted(event_counts.items()):
            print(f"  {event_type}: {count}")

        if verbose:
            print("\n  Event details:")
            for ev in log_events[:20]:  # First 20
                ts = ev.timestamp.strftime("%H:%M:%S.%f")[:12] if ev.timestamp else "??:??:??.???"
                print(f"    [{ts}] {ev.event_type}: {ev.details[:80]}")
            if len(log_events) > 20:
                print(f"    ... and {len(log_events) - 20} more events")

    # Turn analysis
    print("\n--- Turn Analysis ---")
    n_user_segments = len(user_segments)
    n_bot_segments = len(bot_segments)
    print(f"User segments: {n_user_segments}")
    print(f"Bot segments: {n_bot_segments}")

    # Try to pair user/bot segments into turns
    if user_segments and bot_segments:
        print("\nTurn timing (user segment -> bot segment):")
        for i in range(min(n_user_segments, n_bot_segments)):
            user = user_segments[i]
            bot = bot_segments[i]
            gap = bot.start_ms - user.end_ms
            print(f"  Turn {i+1}: User ends at {format_ms(user.end_ms)}, "
                  f"Bot starts at {format_ms(bot.start_ms)} (gap={gap:.0f}ms)")

    # Timing correlation analysis
    if log_events and user_segments and bot_segments:
        print("\n--- Log-to-Audio Timing Correlation ---")
        correlation_results = correlate_log_events_with_audio(
            log_events, user_segments, bot_segments, tolerance_ms=tolerance_ms
        )

        if not correlation_results:
            print("  Could not correlate log events with audio segments")
            print("  (Log events may be missing timestamps)")
        else:
            all_aligned = True
            for result in correlation_results:
                turn = result["turn"]
                if result["mismatches"]:
                    all_aligned = False
                    print(f"\n  Turn {turn} MISMATCHES:")
                    for mismatch in result["mismatches"]:
                        print(f"    - {mismatch}")
                else:
                    user_delta = result.get("user_start_delta_ms", 0)
                    bot_delta = result.get("bot_start_delta_ms", 0)
                    print(f"  Turn {turn}: OK (user_delta={user_delta:.0f}ms, bot_delta={bot_delta:.0f}ms)")

            if all_aligned:
                print("\n  ✓ All segments align within tolerance!")
            else:
                print("\n  ✗ Some segments have timing mismatches")

    # Summary
    print("\n--- Summary ---")
    print(f"Total audio duration: {duration_secs:.2f}s")
    print(f"User speech segments: {n_user_segments}")
    print(f"Bot speech segments: {n_bot_segments}")
    if user_segments and bot_segments:
        total_user_speech = sum(s.duration_ms for s in user_segments) / 1000
        total_bot_speech = sum(s.duration_ms for s in bot_segments) / 1000
        print(f"Total user speech: {total_user_speech:.2f}s")
        print(f"Total bot speech: {total_bot_speech:.2f}s")
        overlap_ratio = (total_user_speech + total_bot_speech) / duration_secs if duration_secs > 0 else 0
        print(f"Speech/silence ratio: {overlap_ratio:.2f}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze conversation.wav for turn segmentation and timing verification."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to run directory containing conversation.wav and run.log"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed log events"
    )
    parser.add_argument(
        "--threshold-db",
        type=float,
        default=-40,
        help="Energy threshold in dB for speech detection (default: -40)"
    )
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=500,
        help="Tolerance in ms for log-to-audio timing correlation (default: 500)"
    )

    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"ERROR: Run directory not found: {args.run_dir}")
        return 1

    analyze_run(args.run_dir, verbose=args.verbose, tolerance_ms=args.tolerance_ms)
    return 0


if __name__ == "__main__":
    exit(main())
