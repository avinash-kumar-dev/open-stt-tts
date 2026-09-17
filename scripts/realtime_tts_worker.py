#!/usr/bin/env python3
"""
Worker for the unofficial supertonic-realtime fork.

Runs ONLY under .venv-realtime (imports as `supertonic`, conflicts with official).
Emits NDJSON lines to stdout for the main PoC web server to forward.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "web"
OUT.mkdir(parents=True, exist_ok=True)


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def word_delta_iter(text: str):
    """Simulate LLM token stream (space-delimited words)."""
    parts = text.split(" ")
    for i, w in enumerate(parts):
        yield w if i == len(parts) - 1 else w + " "


def main() -> int:
    p = argparse.ArgumentParser(description="supertonic-realtime clause stream worker")
    p.add_argument("--text", required=True)
    p.add_argument("--voice", default="M1")
    p.add_argument("--lang", default="en")
    p.add_argument("--speed", type=float, default=1.05)
    p.add_argument("--total-steps", type=int, default=8)
    p.add_argument("--first-chunk-steps", type=int, default=4)
    p.add_argument("--first-chunk-chars", type=int, default=60)
    p.add_argument("--max-chunk-chars", type=int, default=180)
    p.add_argument("--min-chunk-chars", type=int, default=24)
    p.add_argument(
        "--feed",
        choices=("full", "words"),
        default="full",
        help="full string vs simulated LLM word deltas",
    )
    p.add_argument("--custom-style", default=None)
    args = p.parse_args()

    text = (args.text or "").strip()
    if not text:
        emit({"type": "error", "error": "empty text"})
        return 2

    try:
        from supertonic import TTS
    except ImportError as e:
        emit(
            {
                "type": "error",
                "error": f"supertonic-realtime not importable: {e}. "
                "Use .venv-realtime (pip install supertonic-realtime).",
            }
        )
        return 1

    t0 = time.perf_counter()
    tts = TTS(auto_download=True)
    if args.custom_style:
        style = tts.get_voice_style_from_path(args.custom_style)
    else:
        style = tts.get_voice_style(voice_name=args.voice)

    options = {
        "lang": args.lang,
        "speed": args.speed,
        "total_steps": args.total_steps,
        "first_chunk_steps": args.first_chunk_steps,
        "first_chunk_chars": args.first_chunk_chars,
        "max_chunk_chars": args.max_chunk_chars,
        "min_chunk_chars": args.min_chunk_chars,
        "feed": args.feed,
        "voice": args.voice,
    }
    emit(
        {
            "type": "plan",
            "engine": "supertonic-realtime (unofficial fork)",
            "full_text": text,
            "options": options,
            "note": "Yields clause-level AudioChunk (not word/frame). "
            "first_chunk_steps/chars lower TTFB; later clauses use total_steps/max_chunk_chars.",
        }
    )

    source = word_delta_iter(text) if args.feed == "words" else text
    emit({"type": "sending", "text": text if args.feed == "full" else "(word deltas…)", "feed": args.feed})

    first_audio_ms = None
    total_audio_ms = 0.0
    count = 0
    try:
        for chunk in tts.synthesize_stream(
            source,
            style,
            lang=args.lang,
            speed=args.speed,
            total_steps=args.total_steps,
            first_chunk_steps=args.first_chunk_steps,
            first_chunk_chars=args.first_chunk_chars,
            max_chunk_chars=args.max_chunk_chars,
            min_chunk_chars=args.min_chunk_chars,
        ):
            elapsed = (time.perf_counter() - t0) * 1000
            if first_audio_ms is None:
                first_audio_ms = elapsed
            samples = chunk.wav
            import numpy as np
            import soundfile as sf

            arr = np.asarray(samples, dtype=np.float32).reshape(-1)
            out = OUT / f"rtfork_{chunk.index}_{uuid.uuid4().hex[:8]}.wav"
            sf.write(str(out), arr, int(chunk.sample_rate))
            audio_ms = float(chunk.duration_s) * 1000.0
            total_audio_ms += audio_ms
            count += 1
            emit(
                {
                    "type": "audio",
                    "index": chunk.index,
                    "text": chunk.text,
                    "audio_url": f"/out/{out.name}",
                    "sample_rate": int(chunk.sample_rate),
                    "audio_ms": round(audio_ms, 1),
                    "steps_used": int(chunk.total_steps),
                    "elapsed_ms": round(elapsed, 1),
                    "ttfb_ms": round(first_audio_ms, 1),
                    "synth_hint": "clause ready (fork yields after each clause synth)",
                }
            )
    except Exception as e:  # noqa: BLE001
        emit({"type": "error", "error": str(e)})
        return 1

    wall = (time.perf_counter() - t0) * 1000
    emit(
        {
            "type": "done",
            "clause_count": count,
            "ttfb_ms": None if first_audio_ms is None else round(first_audio_ms, 1),
            "wall_ms": round(wall, 1),
            "total_audio_ms": round(total_audio_ms, 1),
            "rtf": round(wall / total_audio_ms, 3) if total_audio_ms > 0 else 0.0,
            "options": options,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
