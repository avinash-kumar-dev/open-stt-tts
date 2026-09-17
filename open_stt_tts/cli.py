"""CLI for open STT / TTS PoC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_stt(args: argparse.Namespace) -> int:
    from .stt_nemotron import stream_mic, stream_wav

    def on_partial(p):
        kind = "final " if p.is_endpoint else "partial"
        print(
            f"[{kind}] t={p.elapsed_ms:.0f}ms audio_fed={p.audio_ms_fed:.0f}ms | {p.text}",
            flush=True,
        )

    if args.mic:
        text, stats = stream_mic(
            language=args.language,
            model_dir=args.model_dir,
            seconds=args.seconds,
            on_partial=on_partial,
        )
    else:
        if not args.wav:
            print("--wav or --mic required", file=sys.stderr)
            return 2
        text, stats = stream_wav(
            args.wav,
            language=args.language,
            chunk_ms=args.chunk_ms,
            model_dir=args.model_dir,
            on_partial=on_partial,
        )

    print("---")
    print(f"FINAL: {text}")
    print(
        json.dumps(
            {
                "partials": stats.partials,
                "endpoints": stats.endpoints,
                "first_partial_ms": stats.first_partial_ms,
                "wall_ms": round(stats.wall_ms, 1),
                "audio_ms": round(stats.audio_ms, 1),
                "rtf": round(stats.rtf, 3),
                "streaming_ok": stats.partials >= 1 and (
                    stats.first_partial_ms is not None
                    and stats.first_partial_ms < stats.audio_ms
                ),
            },
            indent=2,
        )
    )
    return 0


def cmd_tts(args: argparse.Namespace) -> int:
    from .tts_supertonic import synthesize

    def on_chunk(c):
        print(
            f"[chunk {c.index}] synth={c.synth_ms:.0f}ms audio={c.audio_ms:.0f}ms "
            f"rtf={c.rtf:.3f} ttfb_or_cost={c.ttfb_ms:.0f}ms | {c.text[:80]}",
            flush=True,
        )

    out = args.out or "out/tts.wav"
    _, stats = synthesize(
        args.text,
        voice=args.voice,
        lang=args.lang,
        speed=args.speed,
        steps=args.steps,
        custom_style=args.custom_style,
        out=out,
        stream_sentences=args.stream_sentences,
        on_chunk=on_chunk,
    )
    print("---")
    print(
        json.dumps(
            {
                "out": str(out),
                "sample_rate": stats.sample_rate,
                "chunks": len(stats.chunks),
                "ttfb_ms": None if stats.ttfb_ms is None else round(stats.ttfb_ms, 1),
                "wall_ms": round(stats.wall_ms, 1),
                "total_audio_ms": round(stats.total_audio_ms, 1),
                "rtf": round(stats.rtf, 3),
                "stream_sentences": args.stream_sentences,
            },
            indent=2,
        )
    )
    return 0


def cmd_roundtrip(args: argparse.Namespace) -> int:
    """TTS → write wav → resample to 16k → STT streaming."""
    import numpy as np

    from .audio_io import write_wav
    from .stt_nemotron import stream_wav
    from .tts_supertonic import synthesize

    out_tts = Path(args.out or "out/roundtrip_tts.wav")
    print(f"TTS: {args.text!r}")
    samples, tts_stats = synthesize(
        args.text,
        voice=args.voice,
        lang=args.lang,
        out=out_tts,
        stream_sentences=False,
    )
    print(f"TTS done ttfb_ms={tts_stats.ttfb_ms:.0f} rtf={tts_stats.rtf:.3f} -> {out_tts}")

    # STT wants 16 kHz; write a downsampled copy
    out_16k = out_tts.with_name(out_tts.stem + "_16k.wav")
    sr = tts_stats.sample_rate
    if sr != 16000:
        duration = len(samples) / sr
        n = int(duration * 16000)
        x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=max(n, 1), endpoint=False)
        mono = np.interp(x_new, x_old, samples.astype(np.float64)).astype(np.float32)
        write_wav(out_16k, mono, 16000)
    else:
        out_16k = out_tts

    def on_partial(p):
        kind = "final " if p.is_endpoint else "partial"
        print(f"[{kind}] {p.text}", flush=True)

    print(f"STT streaming: {out_16k}")
    text, stt_stats = stream_wav(
        out_16k,
        language=args.language,
        chunk_ms=args.chunk_ms,
        model_dir=args.model_dir,
        on_partial=on_partial,
    )
    print("---")
    print(
        json.dumps(
            {
                "source_text": args.text,
                "stt_text": text,
                "tts_ttfb_ms": round(tts_stats.ttfb_ms or 0, 1),
                "tts_rtf": round(tts_stats.rtf, 3),
                "stt_first_partial_ms": stt_stats.first_partial_ms,
                "stt_rtf": round(stt_stats.rtf, 3),
                "stt_partials": stt_stats.partials,
            },
            indent=2,
        )
    )
    return 0


def cmd_info(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "stt": {
                    "model": "nvidia/nemotron-3.5-asr-streaming-0.6b",
                    "runtime": "sherpa-onnx (ONNX int8 packs)",
                    "streaming": "model-native cache-aware; chunk size fixed per pack",
                    "options": [
                        "language: en|ar|auto|…",
                        "chunk pack: 80|160|320|560|1120 ms via download script",
                        "provider: cpu|cuda",
                        "endpoint detection",
                    ],
                    "download": "./scripts/download_stt_model.sh [560]",
                },
                "tts": {
                    "model": "Supertone/supertonic-3",
                    "runtime": "supertonic (ONNX Runtime)",
                    "streaming": "app/serving layer — sentence chunks + TTFB metric",
                    "options": [
                        "voice: M1–M5, F1–F5",
                        "lang: 31 codes + na",
                        "speed, total_steps, custom voice JSON",
                        "optional: pip install 'supertonic[serve]' && supertonic serve",
                        "optional fork: pip install supertonic-realtime (clause stream)",
                    ],
                    "sample_rate_hz": 44100,
                },
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="open_stt_tts",
        description="Standalone Nemotron STT + Supertonic TTS PoC",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    info = sub.add_parser("info", help="Print model/library options")
    info.set_defaults(func=cmd_info)

    stt = sub.add_parser("stt", help="Streaming STT from wav or mic")
    stt.add_argument("--wav", type=str, default=None)
    stt.add_argument("--mic", action="store_true")
    stt.add_argument("--seconds", type=float, default=0.0, help="Mic duration (0=until Ctrl+C)")
    stt.add_argument("--language", default="auto", help="en | ar | auto | …")
    stt.add_argument("--chunk-ms", type=int, default=100, help="Simulated stream chunk size")
    stt.add_argument("--model-dir", default=None, help="Path to sherpa ONNX pack")
    stt.set_defaults(func=cmd_stt)

    tts = sub.add_parser("tts", help="TTS with TTFB / RTF metrics")
    tts.add_argument("--text", required=True)
    tts.add_argument("--out", default="out/tts.wav")
    tts.add_argument("--voice", default="M1")
    tts.add_argument("--lang", default="en")
    tts.add_argument("--speed", type=float, default=1.0)
    tts.add_argument("--steps", type=int, default=8)
    tts.add_argument(
        "--custom-style",
        default=None,
        help="Path to Voice Builder / custom voice style JSON",
    )
    tts.add_argument(
        "--stream-sentences",
        action="store_true",
        help="Synthesize sentence-by-sentence and report first-chunk TTFB",
    )
    tts.set_defaults(func=cmd_tts)

    rt = sub.add_parser("roundtrip", help="TTS then streaming STT")
    rt.add_argument("--text", required=True)
    rt.add_argument("--out", default="out/roundtrip_tts.wav")
    rt.add_argument("--voice", default="M1")
    rt.add_argument("--lang", default="en")
    rt.add_argument("--language", default="en", help="STT language")
    rt.add_argument("--chunk-ms", type=int, default=100)
    rt.add_argument("--model-dir", default=None)
    rt.set_defaults(func=cmd_roundtrip)

    serve = sub.add_parser("serve", help="Open clickable web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=cmd_serve)

    return p


def cmd_serve(args: argparse.Namespace) -> int:
    from .web import main as serve_main

    serve_main(host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
